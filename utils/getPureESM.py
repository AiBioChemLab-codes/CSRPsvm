import sys
import os
import time
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AlbertModel, AlbertTokenizer, 
    BertModel, BertTokenizer,
    T5EncoderModel, T5Tokenizer,
    AutoTokenizer, AutoModel
)
from tqdm.auto import tqdm
import psutil
import gc
import pandas as pd
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import os
import tempfile
import torch

# Setup PyTorch cache directory
def setup_pytorch_cache():
    # Method 1: Use a user-writable temporary directory
    cache_dir = os.path.join(tempfile.gettempdir(), f"torch_cache_{os.getuid()}")
    os.makedirs(cache_dir, exist_ok=True)
    os.chmod(cache_dir, 0o777)  # Ensure read/write permissions
    
    # Set PyTorch cache environment variables
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir
    os.environ['TRITON_CACHE_DIR'] = cache_dir
    os.environ['TORCH_EXTENSIONS_DIR'] = cache_dir
    
    # Clean old cache
    torch.cuda.empty_cache()
    
    return cache_dir

# Call at the beginning of code
cache_dir = setup_pytorch_cache()
print(f"PyTorch cache directory set to: {cache_dir}")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(BASE_DIR,"utils")

# Global time tracker dictionary
time_tracker = {}

# Multi-GPU management class
class GPUMemoryManager:
    def __init__(self):
        self.devices = self.get_available_devices()
        self.print_gpu_info()
    
    def get_available_devices(self):
        """Get information about all available GPU devices"""
        devices = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                total_mem = torch.cuda.get_device_properties(i).total_memory
                free_mem = total_mem - torch.cuda.memory_allocated(device)
                devices.append({
                    "device": device,
                    "id": i,
                    "total_mem": total_mem,
                    "free_mem": free_mem,
                    "name": torch.cuda.get_device_name(i)
                })
        
        # Always add CPU device
        devices.append({
            "device": torch.device("cpu"),
            "id": -1,
            "total_mem": psutil.virtual_memory().total,
            "free_mem": psutil.virtual_memory().available,
            "name": "CPU"
        })
        return devices
    
    def print_gpu_info(self):
        """Print device information"""
        print("\n" + "="*50)
        print("Available Devices:")
        for i, device in enumerate(self.devices):
            if device['id'] != -1:  # GPU device
                print(f"GPU {device['id']}: {device['name']} | Free Mem: {device['free_mem']/1024**3:.2f}GB / {device['total_mem']/1024**3:.2f}GB")
            else:  # CPU device
                print(f"CPU: {device['name']} | Free RAM: {device['free_mem']/1024**3:.2f}GB / {device['total_mem']/1024**3:.2f}GB")
        print("="*50 + "\n")
    
    def select_best_device(self, required_mem=0, use_gpu=False):
        """
        Select the best device based on required memory.
        required_mem: required memory size (bytes)
        use_gpu: whether to use GPU, default False (use CPU)
        Strategy:
        1. If use_gpu is False, directly use CPU.
        2. If use_gpu is True, check if GPU is available.
        3. If GPU is available, check if a single GPU has enough memory.
        4. If a single GPU is insufficient, try multi-GPU.
        5. If multi-GPU is also insufficient or no GPU is available, use CPU.
        """
        # Get CPU device
        cpu_device = [d for d in self.devices if d['id'] == -1][0]
        
        # 1. If not using GPU, directly return CPU
        if not use_gpu:
            print("Info: GPU usage disabled by user. Using CPU.")
            return cpu_device["device"]
        
        # 2. Get GPU devices (exclude CPU)
        gpu_devices = [d for d in self.devices if d['id'] != -1]
        
        # 3. No available GPU, fallback to CPU
        if not gpu_devices:
            print("Info: No GPU available. Using CPU.")
            return cpu_device["device"]
        
        # 4. Sort GPUs by free memory descending
        sorted_gpus = sorted(gpu_devices, key=lambda x: x["free_mem"], reverse=True)
        
        # 5. Check if any single GPU meets the requirement
        for gpu in sorted_gpus:
            if gpu["free_mem"] > required_mem * 1.2:
                print(f"Selected GPU {gpu['id']} ({gpu['name']}) with {gpu['free_mem']/1024**3:.2f}GB free memory")
                return gpu["device"]
        
        # 6. No single GPU meets requirement, try multi-GPU
        if len(sorted_gpus) >= 2:
            selected_gpus = sorted_gpus[:2]
            total_free_mem = sum(gpu["free_mem"] for gpu in selected_gpus)
            if total_free_mem > required_mem * 1.2:
                print(f"Using multi-GPU setup on devices {[gpu['id'] for gpu in selected_gpus]}")
                return [gpu["device"] for gpu in selected_gpus]
        
        # 7. GPU memory insufficient, fallback to CPU
        print(f"Warning: GPU memory insufficient. Required: {required_mem/1024**3:.2f}GB. Falling back to CPU.")
        return cpu_device["device"]
    
    def setup_multi_gpu(self, required_mem):
        """Set up multi-GPU parallelism (called by select_best_device)"""
        gpu_devices = [d for d in self.devices if d['id'] != -1]
        sorted_gpus = sorted(gpu_devices, key=lambda x: x["free_mem"], reverse=True)
        
        if len(sorted_gpus) < 2:
            print("Info: Not enough GPUs for multi-GPU setup.")
            return None
        
        selected_gpus = sorted_gpus[:2]
        total_free_mem = sum(gpu["free_mem"] for gpu in selected_gpus)
        
        if total_free_mem < required_mem * 1.2:
            print(f"Warning: Combined GPU memory insufficient. Required: {required_mem/1024**3:.2f}GB, Available: {total_free_mem/1024**3:.2f}GB")
            return None
        
        print(f"Using multi-GPU setup on devices {[gpu['id'] for gpu in selected_gpus]}")
        return [gpu["device"] for gpu in selected_gpus]

# Global GPU manager
gpu_manager = GPUMemoryManager()

def print_gpu_memory():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
    else:
        print("Using CPU, GPU not available")

def clear_gpu_memory(*objects):
    """Clear GPU memory objects and reclaim resources, also works for CPU"""
    for obj in objects:
        if obj is not None:
            del obj
    gc.collect()  # Force garbage collection, release all unreferenced objects (CPU and GPU)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Only clear CUDA cache when using GPU
    print("Memory cleared (CPU RAM and GPU cache if applicable)")

def estimate_model_size(model_class, model_path):
    """Estimate model size (bytes)"""
    model_size_map = {
        
        "esm2_t33_650M_UR50D": 3 * 1024**3,
        
    }
    
    model_name = model_path.split("/")[-1] if "/" in model_path else model_path
    return model_size_map.get(model_name, 2 * 1024**3)

# ====================== Accelerated Feature Extraction Core ======================
class AcceleratedFeatureExtractor:
    def __init__(self, model, tokenizer, device, max_seq_len=2000):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        # Enable compilation acceleration only on non-CPU
        if hasattr(torch, "compile") and device.type != 'cpu':
            self.model = torch.compile(model)
        
        # Enable mixed precision computation (only available on GPU)
        self.use_amp = isinstance(device, torch.device) and device.type == 'cuda' and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        self.scaler = torch.amp.GradScaler() if self.use_amp else None
    
    def process_batch(self, sequences, order_ids, pids, batch_size=8):
        """Efficient batch processing logic"""
        # Create dataset
        dataset = EmbeddingDataset(sequences, order_ids, self.tokenizer, self.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=min(4, os.cpu_count()//2),  # Optimized data loading
                               pin_memory=True)
        
        all_embeddings = []
        for batch in tqdm(dataloader, desc="Processing batches"):
            
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            seq_lens = batch['seq_len'].numpy()
            
            # Initialize variables
            outputs = None
            embeddings = None
            
            try:
                with torch.no_grad():
                    # Use latest AMP API 
                    if self.use_amp and isinstance(self.device, torch.device) and self.device.type == 'cuda':
                        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):  # Mixed precision acceleration
                            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Process outputs
                embeddings = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                
                # Batch feature extraction
                for i in range(embeddings.size(0)):
                    seq_emb = embeddings[i, :seq_lens[i]]
                    mean_emb = seq_emb.mean(dim=0).cpu().numpy()
                    all_embeddings.append(mean_emb)
            
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"Warning: Batch OOM (size {batch_size}), processing sequences individually")
                    # Single sequence fallback
                    for i in range(len(input_ids)):
                        self.process_single_sequence(input_ids[i:i+1], attention_mask[i:i+1], seq_lens[i], all_embeddings)
                else:
                    raise
            finally:
                # Clear memory
                if outputs is not None:
                    del outputs
                if embeddings is not None:
                    del embeddings
                
                del input_ids, attention_mask
                # Garbage collect after each batch regardless of CPU or GPU
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Build result DataFrame
        if all_embeddings:
            df = pd.DataFrame(all_embeddings)
            actual_feat_size = len(df.columns)
            df.columns = [f"{self.model.config.model_type}_F{i+1}" for i in range(actual_feat_size)]
            df = pd.concat([pd.Series(order_ids, name="Order_ID"), df], axis=1)
            df.index = pids
            return df
        return None
    
    def process_single_sequence(self, input_ids, attention_mask, seq_len, embeddings_list):
        """Emergency fallback for single sequence processing"""
        try:
            with torch.no_grad():
                # Use latest AMP API 
                if self.use_amp and isinstance(self.device, torch.device) and self.device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        outputs = self.model(input_ids=input_ids.unsqueeze(0), 
                                           attention_mask=attention_mask.unsqueeze(0))
                else:
                    outputs = self.model(input_ids=input_ids.unsqueeze(0), 
                                       attention_mask=attention_mask.unsqueeze(0))
                
                embedding = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
                seq_emb = embedding[0, :seq_len]
                mean_emb = seq_emb.mean(dim=0).cpu().numpy()
                embeddings_list.append(mean_emb)
                
                # Clear memory
                del outputs, embedding, seq_emb
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Error: Sequence too long (len={seq_len}), skipping")
            else:
                raise
        finally:
            # Garbage collect after single sequence processing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# ====================== Optimized Feature Extraction Function ======================
def ESM_Embed(seq_pd, model_type="m7", batch_size=8, max_seq_len=2000, device=None):
    """Optimized ProtTrans feature extraction (2-5x speedup)
    
    Args:
        seq_pd: DataFrame containing sequence data
        model_type: model type, from m1 to m10
        batch_size: batch size
        max_seq_len: maximum sequence length
        device: specify device (torch.device or string like 'cuda:0' or 'cpu'), if None auto-select
    """
    start_time = time.time()
    model_name = ""
    print("\n" + "="*50)
    print(f"ACCELERATED ProtTrans Embedding (Type: {model_type})")
    if device is not None:
        print(f"Device: {device}")
    else:
        print(f"Device: auto-detect")
    print("="*50)
    print_gpu_memory()
    
    # Simplified model configuration
    MODEL_CONFIG = {
         
        "m7": ("esm2_t33_650M_UR50D", AutoModel, AutoTokenizer,  os.path.join(BASE_DIR,"ESM2","esm2_t33_650M_UR50D")),
         
    }

    if model_type not in MODEL_CONFIG:
        raise ValueError(f"Invalid model type: {model_type}. Choose between m6-m10.")
    
    model_name, model_class, tokenizer_class, model_path = MODEL_CONFIG[model_type]
    print(f"Using {model_name} from {model_path}")
    
    # Special tokenizer arguments
    tokenizer_args = {}
    
    # Initialize variables to avoid errors in finally block
    tokenizer = None
    model = None
    
    try:
        # Load tokenizer - fix: add local_files_only parameter
        try:
            tokenizer = tokenizer_class.from_pretrained(model_path, local_files_only=True, **tokenizer_args)
        except Exception as e:
            print(f"Warning: Failed to load tokenizer with local_files_only=True: {e}")
            print("Info: Trying to load tokenizer without local_files_only...")
            tokenizer = tokenizer_class.from_pretrained(model_path, **tokenizer_args)
        
        # Determine device
        if device is None:
            # Auto-select device
            model_size = estimate_model_size(model_class, model_path)
            device = gpu_manager.select_best_device(model_size, use_gpu=True)  # Default attempt to use GPU
        else:
            # Convert string to torch.device
            if isinstance(device, str):
                device = torch.device(device)
            print(f"Info: Using specified device: {device}")
        
        # Load model - fix: add local_files_only parameter
        try:
            model = model_class.from_pretrained(model_path, local_files_only=True)
        except Exception as e:
            print(f"Warning: Failed to load model with local_files_only=True: {e}")
            print("Info: Trying to load model without local_files_only...")
            model = model_class.from_pretrained(model_path)
        
        # Handle multi-GPU case
        if isinstance(device, list):
            print(f"Using DataParallel across {len(device)} GPUs")
            model = nn.DataParallel(model, device_ids=[d.index for d in device])
            device = device[0]
        
        model.to(device)
        model.eval()
        print(f"Model loaded. Memory after loading:")
        print_gpu_memory()
        
        # Prepare data
        order_ids, sequences, pids = tokenize(seq_pd)
        
        # Sequence analysis
        avg_len = sum(map(len, sequences)) / len(sequences)
        print(f"Sequences: {len(sequences)}, Avg length: {avg_len:.1f}")
        
        # Use accelerated processor
        extractor = AcceleratedFeatureExtractor(model, tokenizer, device, max_seq_len)
        df = extractor.process_batch(sequences, order_ids, pids, batch_size)
    
    finally:
        # Regardless of success, clean up memory
        clear_gpu_memory(model, tokenizer)
        # Extra garbage collection to ensure CPU memory is freed
        gc.collect()
        if device is not None and (isinstance(device, torch.device) and device.type == 'cpu') or (isinstance(device, str) and 'cpu' in device):
            print("CPU memory cleanup completed after feature extraction.")
    
    # Record time taken
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    time_tracker[model_name] = elapsed_minutes
    
    if df is not None:
        print(f"{model_name} embeddings completed in {elapsed_minutes:.2f} mins")
        print_gpu_memory()
        print(df.shape)
        return df
    
    print(f"Failed to generate embeddings for {model_name}")
    return None

# ====================== Helper Functions ======================
def tokenize(data_frame):
    """Prepare data for embedding"""
    order_ids = data_frame["Order_ID"]
    sequences = data_frame["Sequence"]
    pids = data_frame["ID"]
    return np.array(order_ids), np.array(sequences), np.array(pids)

class EmbeddingDataset(Dataset):
    """Efficient dataset class"""
    def __init__(self, sequences, order_ids, tokenizer, max_length=1000):
        self.sequences = sequences
        self.order_ids = order_ids
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequence_lengths = [len(seq) for seq in sequences]
        # Pre-clean sequences
        self.cleaned_sequences = [re.sub(r"[1234567890JUZOB]", "X", seq) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.cleaned_sequences[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'order_ids': self.order_ids[idx],
            'seq_len': min(self.sequence_lengths[idx], self.max_length)
        }

def print_time_summary():
    """Print feature extraction time summary table"""
    if not time_tracker:
        print("No feature extraction tasks completed")
        return
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION TIME SUMMARY")
    print("="*70)
    
    total_time = 0
    fastest_model = None
    slowest_model = None
    min_time = float('inf')
    max_time = 0
    
    # Print table header
    print(f"{'Model':<25}{'Time (mins)':<15}{'Percentage':<15}")
    print("-"*55)
    
    # Calculate total time
    total_time = sum(time_tracker.values())
    
    # Print time for each model
    for model, t in time_tracker.items():
        percentage = (t / total_time) * 100 if total_time > 0 else 0
        print(f"{model:<25}{t:.2f} min{percentage:>10.1f}%")
        
        # Record fastest and slowest models
        if t < min_time:
            min_time = t
            fastest_model = model
        if t > max_time:
            max_time = t
            slowest_model = model
    
    # Calculate average time
    avg_time = total_time / len(time_tracker) if time_tracker else 0
    
    # Print statistics summary
    print("\nTime Statistics:")
    print(f"Total models processed: {len(time_tracker)}")
    print(f"Total time spent: {total_time:.2f} min")
    print(f"Average time per model: {avg_time:.2f} min")
    print(f"Fastest model: {fastest_model} ({min_time:.2f} min)")
    print(f"Slowest model: {slowest_model} ({max_time:.2f} min)")
    
    # Print optimization tips
    print("\nTips for optimization:")
    if max_time > 2 * avg_time:
        print(f"- Consider optimizing {slowest_model} as it's taking disproportionately long")
    if min_time < 0.5:
        print(f"- {fastest_model} is very efficient, consider using it for quick results")
    
    print("="*70 + "\n")