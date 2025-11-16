"""
Training Script for Tunix GRPO
Loads cached tokenized data and trains with GRPO
Updated to handle single dataset file with train/test split
"""

import functools
import gc
import os
import pickle
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List

import grain
import humanize
import jax
import jax.numpy as jnp
import jax.sharding as shd
import optax
from orbax import checkpoint as ocp
from tqdm.auto import tqdm

try:
    from jax.extend import backend as jax_backend  # JAX >= 0.8.0

    def _get_jax_backend():
        return jax_backend.get_backend()

except (ImportError, AttributeError):
    from jax.lib import xla_bridge  # Older JAX releases

    def _get_jax_backend():
        return xla_bridge.get_backend()

from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as params_lib
from tunix.models.gemma import params_safetensors as params_safetensors_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.sft import metrics_logger
import numpy as np

try:
    import qwix
except ImportError:  # pragma: no cover
    qwix = None


def _build_default_mesh(devices: List[jax.Device]):
    """Create a simple (fsdp, tp) mesh covering all devices."""
    if not devices:
        return None
    num_devices = len(devices)
    mesh_shape = (1, num_devices) if num_devices > 1 else (1, 1)
    axis_names = ("fsdp", "tp")
    try:
        device_array = np.array(devices, dtype=object).reshape(mesh_shape)
    except ValueError as exc:
        warnings.warn(
            f"Unable to create mesh with shape {mesh_shape}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return shd.Mesh(device_array, axis_names)


# ====== CONFIGURATION ======
class Config:
    # Model - Gemma 2 2B
    MODEL_ID = "google/gemma-2-2b-it"
    GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer.model"
    MODEL_VERSION = "2-2b-it"
    
    # Data
    DATASET_FILE = "/kaggle/input/tunix-comp-test-data/cached_data/gsm8k.pkl"  # Your single dataset file
    TRAIN_FRACTION = 0.9  # 90% for training, 10% for validation
    
    # LoRA
    RANK = 64
    ALPHA = 64.0
    LORA_MODULE_PATH = (
        ".*q_einsum|.*kv_einsum|.*qkv_einsum|.*gate_proj|"
        ".*down_proj|.*up_proj|.*attn_vec_einsum"
    )
    
    # Sharding / hardware awareness
    JAX_DEVICES = jax.devices()
    NUM_DEVICES = len(JAX_DEVICES)
    JAX_PLATFORM = _get_jax_backend().platform if JAX_DEVICES else "unknown"
    MODEL_MESH = _build_default_mesh(JAX_DEVICES)
    
    if NUM_DEVICES == 0:
        raise RuntimeError(
            "No JAX devices detected. Ensure that jax is installed with the desired backend."
        )
    
    # GRPO Generation
    MAX_PROMPT_LENGTH = 256
    TOTAL_GENERATION_STEPS = 768
    TEMPERATURE = 0.9
    TOP_P = 1.0
    TOP_K = 50
    NUM_GENERATIONS = 2
    
    # GRPO Training
    NUM_ITERATIONS = 1
    BETA = 0.08
    EPSILON = 0.2
    
    # Training
    TRAIN_MICRO_BATCH_SIZE = 1
    EVAL_EVERY_N_STEPS = 64
    NUM_EPOCHS = 1
    
    # Optimizer
    LEARNING_RATE = 3e-6
    B1 = 0.9
    B2 = 0.99
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS_RATIO = 0.1
    MAX_GRAD_NORM = 0.1
    
    # Checkpointing
    INTERMEDIATE_CKPT_DIR = "/tmp/content/intermediate_ckpt/"
    CKPT_DIR = "/tmp/content/ckpts/"
    SAVE_INTERVAL_STEPS = 500
    MAX_TO_KEEP = 4
    
    # Inference
    GENERATION_CONFIGS = {
        "greedy": {"temperature": None, "top_k": 1, "top_p": None},
        "standard": {"temperature": 0.7, "top_k": 50, "top_p": 0.95},
        "liberal": {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
    }


# ====== UTILITY FUNCTIONS ======
def show_hbm_usage():
    """Display memory usage per device."""
    fmt_size = functools.partial(humanize.naturalsize, binary=True)
    
    for d in jax.local_devices():
        stats = d.memory_stats()
        used = stats["bytes_in_use"]
        limit = stats["bytes_limit"]
        print(f"Using {fmt_size(used)} / {fmt_size(limit)} ({used/limit:%}) on {d}")


def load_and_split_dataset(
    cache_file: str, 
    train_fraction: float = 0.9,
    seed: int = 42
) -> tuple[List[Dict], List[Dict], Dict]:
    """
    Load cached dataset and split into train/test.
    
    Args:
        cache_file: Path to the .pkl file
        train_fraction: Fraction of data to use for training
        seed: Random seed for splitting
        
    Returns:
        train_data, test_data, metadata
    """
    print(f"Loading dataset from {cache_file}...")
    
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Cache file not found: {cache_file}\n"
            f"Please run data_preprocess.py first!\n"
            f"Example: python data_preprocess.py --input data/train.json --output gsm8k.pkl"
        )
    
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    
    data = cache["data"]
    metadata = cache.get("metadata", {})
    
    print(f"✓ Loaded {len(data)} total examples")
    
    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(data))
    rng.shuffle(indices)
    
    split_idx = int(len(data) * train_fraction)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    
    print(f"  Train examples: {len(train_data)} ({train_fraction*100:.0f}%)")
    print(f"  Test examples: {len(test_data)} ({(1-train_fraction)*100:.0f}%)")
    
    if metadata:
        print(f"  Metadata: {metadata}")
    
    return train_data, test_data, metadata


def create_grain_dataset(
    cached_data: List[Dict],
    shuffle: bool = True,
    seed: int = 42
) -> grain.MapDataset:
    """Create a Grain dataset from cached data."""
    
    dataset = grain.MapDataset.source(cached_data)
    
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    
    # The data is already processed, so we just pass it through
    dataset = dataset.map(lambda x: {
        "prompts": x["prompt"],
        "question": x["question"],
        "answer": x["answer"],
    })
    
    return dataset


# ====== REWARD FUNCTIONS ======
def create_reward_functions():
    """
    Create reward functions for GRPO training.
    
    These functions evaluate how good the model's response is.
    Higher reward = better response.
    """
    
    def has_reasoning_tags(response: str) -> float:
        """Reward for including reasoning tags."""
        has_start = "<reasoning>" in response
        has_end = "</reasoning>" in response
        has_both = has_start and has_end
        return 1.0 if has_both else 0.0
    
    def has_answer_tags(response: str) -> float:
        """Reward for including answer tags."""
        has_start = "<answer>" in response
        has_end = "</answer>" in response
        has_both = has_start and has_end
        return 1.0 if has_both else 0.0
    
    def correct_format(response: str) -> float:
        """Reward for following the complete format."""
        reasoning_score = has_reasoning_tags(response)
        answer_score = has_answer_tags(response)
        
        # Bonus if reasoning comes before answer
        if reasoning_score > 0 and answer_score > 0:
            reasoning_idx = response.find("<reasoning>")
            answer_idx = response.find("<answer>")
            order_bonus = 0.5 if reasoning_idx < answer_idx else 0.0
            return reasoning_score + answer_score + order_bonus
        
        return reasoning_score + answer_score
    
    def reasoning_length(response: str) -> float:
        """Reward for having substantial reasoning (not too short, not too long)."""
        match = re.search(r'<reasoning>(.*?)</reasoning>', response, re.DOTALL)
        if not match:
            return 0.0
        
        reasoning = match.group(1).strip()
        words = len(reasoning.split())
        
        # Optimal range: 20-200 words
        if words < 10:
            return 0.0
        elif words < 20:
            return words / 20.0
        elif words <= 200:
            return 1.0
        else:
            # Penalize if too long
            return max(0.0, 1.0 - (words - 200) / 200)
    
    def extract_model_answer(response: str) -> str:
        """Extract answer from response."""
        match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def answer_correctness(
        response: str,
        ground_truth: str
    ) -> float:
        """
        Reward for correct answer.
        For GSM8K, answers are typically numbers.
        """
        if not ground_truth:
            return 0.0
        
        model_answer = extract_model_answer(response)
        
        # Extract numbers from both answers
        def extract_number(text):
            # Remove commas and extract number
            text = text.replace(",", "")
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                try:
                    return float(match.group())
                except:
                    return None
            return None
        
        model_num = extract_number(model_answer)
        truth_num = extract_number(str(ground_truth))
        
        if model_num is not None and truth_num is not None:
            # Exact match
            if abs(model_num - truth_num) < 0.001:
                return 2.0
            # Close enough (within 1%)
            elif truth_num != 0 and abs(model_num - truth_num) / abs(truth_num) < 0.01:
                return 1.0
        
        # Fallback to string matching
        if model_answer.lower() == str(ground_truth).lower():
            return 2.0
        if str(ground_truth).lower() in model_answer.lower():
            return 1.0
        
        return 0.0
    
    # Combine reward functions
    def combined_reward(response: str, question: str, answer: str) -> float:
        """Main reward function combining all criteria."""
        format_reward = correct_format(response)
        length_reward = reasoning_length(response)
        correctness_reward = answer_correctness(response, answer)
        
        # Weighted combination
        total = (
            format_reward * 1.0 +      # Format is important
            length_reward * 0.5 +       # Decent reasoning
            correctness_reward * 2.0    # Correctness is most important!
        )
        
        return total
    
    return combined_reward


def apply_lora_if_requested(model: gemma_lib.Transformer):
    """Apply LoRA adapters if a positive rank is configured."""
    if Config.RANK <= 0:
        return model
    if qwix is None:
        print("  ⚠️ LoRA requested but `qwix` is not installed; skipping LoRA.")
        return model
    print(
        f"  Applying LoRA adapters (rank={Config.RANK}, alpha={Config.ALPHA})..."
    )
    lora_provider = qwix.LoraProvider(
        module_path=Config.LORA_MODULE_PATH,
        rank=Config.RANK,
        alpha=Config.ALPHA,
    )
    model_inputs = model.get_model_input()
    return qwix.apply_lora_to_model(model, lora_provider, **model_inputs)


def load_model_from_checkpoint(model_path: str) -> tuple[gemma_lib.Transformer, str]:
    """Load Gemma2 weights from either Orbax checkpoints or safetensors files."""
    metadata_file = Path(model_path) / "_METADATA"
    mesh = Config.MODEL_MESH
    if metadata_file.exists():
        print("  Found Orbax checkpoint metadata. Using StandardCheckpointer...")
        params = params_lib.load_and_format_params(str(model_path))
        base_model = gemma_lib.Transformer.from_params(
            params,
            version=Config.MODEL_VERSION,
        )
        del params
        gc.collect()
        load_format = "Orbax checkpoint"
    else:
        print("  No Orbax metadata found. Loading safetensors weights...")
        try:
            model_config = gemma_lib.ModelConfig.gemma2_2b_it()
        except AttributeError:
            model_config = gemma_lib.ModelConfig.gemma2_2b()
        base_model = params_safetensors_lib.create_model_from_safe_tensors(
            file_dir=model_path,
            config=model_config,
            mesh=mesh,
        )
        load_format = "safetensors weights"
    model = apply_lora_if_requested(base_model)
    return model, load_format


# ====== MAIN TRAINING FUNCTION ======
def main():
    print("="*60)
    print("TUNIX GRPO TRAINING - GSM8K")
    print("="*60)
    print(f"\nJAX devices: {Config.JAX_DEVICES}")
    print(f"Detected platform: {Config.JAX_PLATFORM}")
    print(f"Number of devices: {Config.NUM_DEVICES}")
    
    # Load tokenizer
    print("\n1. Loading Gemma 2 2B tokenizer from HuggingFace...")
    from transformers import AutoTokenizer
    
    hf_tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_ID)
    
    # Wrap HuggingFace tokenizer for Tunix
    class HFTokenizerWrapper:
        def __init__(self, hf_tok):
            self.hf_tok = hf_tok
            self.vocab_size = len(hf_tok)
        
        def encode(self, text):
            return self.hf_tok.encode(text, add_special_tokens=False)
        
        def decode(self, tokens):
            return self.hf_tok.decode(tokens)
    
    tokenizer = HFTokenizerWrapper(hf_tokenizer)
    print(f"✓ Tokenizer loaded (vocab size: {tokenizer.vocab_size})")
    print(f"  Model: {Config.MODEL_ID}")
    
    # Load and split dataset
    print("\n2. Loading and splitting dataset...")
    train_data, test_data, metadata = load_and_split_dataset(
        Config.DATASET_FILE,
        train_fraction=Config.TRAIN_FRACTION
    )
    
    # Create Grain datasets
    print("\n3. Creating Grain datasets...")
    train_dataset = create_grain_dataset(train_data, shuffle=True)
    test_dataset = create_grain_dataset(test_data, shuffle=False)
    
    # Calculate number of steps
    num_train_examples = len(train_data)
    NUM_BATCHES = num_train_examples // Config.TRAIN_MICRO_BATCH_SIZE
    MAX_STEPS = int(NUM_BATCHES * Config.NUM_ITERATIONS * Config.NUM_EPOCHS)
    WARMUP_STEPS = int(Config.WARMUP_STEPS_RATIO * MAX_STEPS)
    
    print(f"  Training examples: {num_train_examples}")
    print(f"  Test examples: {len(test_data)}")
    print(f"  Batches per epoch: {NUM_BATCHES}")
    print(f"  Total training steps: {MAX_STEPS}")
    print(f"  Warmup steps: {WARMUP_STEPS}")
    
    # Download and load model
    print("\n4. Downloading Gemma 2 2B model from HuggingFace...")
    from huggingface_hub import snapshot_download
    
    model_path = snapshot_download(
        repo_id=Config.MODEL_ID,
        allow_patterns=["*.safetensors", "*.json"]
    )
    print(f"✓ Model downloaded to: {model_path}")
    
    print("\n5. Building Gemma 2 2B model...")
    model, model_format = load_model_from_checkpoint(model_path)
    print(f"✓ Model initialized from {model_format}")
    
    show_hbm_usage()
    
    # Create optimizer
    print("\n6. Creating optimizer with warmup + cosine decay...")
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=Config.LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=MAX_STEPS,
        end_value=0.0,
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(Config.MAX_GRAD_NORM),
        optax.adamw(
            learning_rate=schedule,
            b1=Config.B1,
            b2=Config.B2,
            weight_decay=Config.WEIGHT_DECAY,
        ),
    )
    print("✓ Optimizer created")
    
    # Create reward function
    print("\n7. Creating reward function...")
    reward_fn = create_reward_functions()
    print("✓ Reward function created")
    print("   Rewards: format (1.0) + length (0.5) + correctness (2.0)")
    
    # Create GRPO config
    grpo_config = GRPOConfig(
        num_iterations=Config.NUM_ITERATIONS,
        beta=Config.BETA,
        epsilon=Config.EPSILON,
        num_generations=Config.NUM_GENERATIONS,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        top_k=Config.TOP_K,
        max_generate_steps=Config.TOTAL_GENERATION_STEPS,
    )
    
    # Create GRPO learner
    print("\n8. Creating GRPO learner...")
    learner = GRPOLearner(
        model=model,
        optimizer=optimizer,
        config=grpo_config,
        reward_fn=reward_fn,
        tokenizer=tokenizer,
    )
    print("✓ GRPO learner created")
    
    # Create checkpoint manager
    print("\n9. Setting up checkpointing...")
    os.makedirs(Config.CKPT_DIR, exist_ok=True)
    os.makedirs(Config.INTERMEDIATE_CKPT_DIR, exist_ok=True)
    
    checkpointer = ocp.CheckpointManager(
        Config.CKPT_DIR,
        options=ocp.CheckpointManagerOptions(
            max_to_keep=Config.MAX_TO_KEEP,
            save_interval_steps=Config.SAVE_INTERVAL_STEPS,
        ),
    )
    print("✓ Checkpoint manager ready")
    print(f"   Saving every {Config.SAVE_INTERVAL_STEPS} steps to {Config.CKPT_DIR}")
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    # Print a sample prompt before training
    print("\n" + "="*60)
    print("EXAMPLE TRAINING PROMPT:")
    print("="*60)
    sample_iter = iter(train_dataset)
    sample = next(sample_iter)
    print(f"\n📝 Prompt:")
    print(sample['prompts'][:400] + "..." if len(sample['prompts']) > 400 else sample['prompts'])
    print(f"\n❓ Question: {sample['question'][:200]}")
    print(f"\n✅ Expected Answer: {sample['answer']}")
    print("="*60 + "\n")
    
    metrics_log = metrics_logger.MetricsLogger()
    
    step = 0
    best_reward = -float('inf')
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{Config.NUM_EPOCHS}")
        print(f"{'='*60}")
        
        data_iter = iter(train_dataset)
        
        with tqdm(total=NUM_BATCHES, desc=f"Epoch {epoch+1}") as pbar:
            for batch_idx in range(NUM_BATCHES):
                # Get batch
                try:
                    batch = [next(data_iter) for _ in range(Config.TRAIN_MICRO_BATCH_SIZE)]
                except StopIteration:
                    print("\n⚠️  Reached end of dataset early, breaking...")
                    break
                
                # Train step
                metrics = learner.train_step(batch)
                
                # Log metrics
                metrics_log.log(metrics)
                
                # Track best reward
                avg_reward = metrics.get('avg_reward', 0.0)
                if avg_reward > best_reward:
                    best_reward = avg_reward
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{metrics.get('loss', 0.0):.4f}",
                    "reward": f"{avg_reward:.2f}",
                    "best_reward": f"{best_reward:.2f}",
                    "kl": f"{metrics.get('kl_div', 0.0):.4f}",
                })
                pbar.update(1)
                
                step += 1
                
                # Save checkpoint
                if step % Config.SAVE_INTERVAL_STEPS == 0:
                    print(f"\n💾 Saving checkpoint at step {step}...")
                    checkpointer.save(step, model)
                    print("✓ Checkpoint saved")
                    
                    # Print a sample generation
                    print("\n" + "="*60)
                    print(f"SAMPLE AT STEP {step}:")
                    print("="*60)
                    sample_iter_gen = iter(train_dataset)
                    sample_gen = next(sample_iter_gen)
                    print(f"\n📝 Question: {sample_gen['question'][:200]}")
                    print(f"✅ Answer: {sample_gen['answer']}")
                    print("="*60 + "\n")
                
                # Show memory usage periodically
                if step % 100 == 0:
                    print("\n")
                    show_hbm_usage()
                    print()
                
                # Run garbage collection periodically
                if step % 50 == 0:
                    gc.collect()
    
    # Final checkpoint
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\n💾 Saving final checkpoint...")
    checkpointer.save(step, model)
    print("✓ Final checkpoint saved")
    
    # Save metrics
    metrics_path = os.path.join(Config.CKPT_DIR, "metrics.json")
    metrics_log.save(metrics_path)
    print(f"✓ Metrics saved to {metrics_path}")
    
    print("\n🎉 Training finished successfully!")
    print(f"   Total steps: {step}")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"   Checkpoints: {Config.CKPT_DIR}")
    
    show_hbm_usage()


if __name__ == "__main__":
    main()


# ====== USAGE ======
"""
1. First, preprocess your GSM8K data:
   python data_preprocess.py --input data/gsm8k_train.json --output gsm8k.pkl

2. Then run training:
   python train_grpo.py

3. The script will automatically:
   - Load gsm8k.pkl
   - Split it 90% train / 10% test
   - Train with GRPO
   - Save checkpoints every 500 steps
   - Log metrics

4. Customize in Config class:
   - DATASET_FILE: path to your .pkl file
   - TRAIN_FRACTION: train/test split ratio
   - RANK/ALPHA: LoRA settings
   - LEARNING_RATE: optimizer settings
   - NUM_EPOCHS: training duration

5. Monitor:
   - Watch progress bar for loss, reward, KL
   - Checkpoints in /tmp/content/ckpts/
   - Metrics in metrics.json
"""
