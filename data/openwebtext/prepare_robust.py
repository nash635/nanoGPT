#!/usr/bin/env python3
"""
Robust OpenWebText dataset preparation script with automatic retry and resume capability.
This script handles network interruptions and allows resuming from where it left off.
"""

import os
import json
import time
import signal
import sys
from functools import wraps
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

def retry_on_failure(max_retries=3, delay=5):
    """Decorator to retry function calls on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

class DatasetPreparer:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or os.path.dirname(__file__)
        self.checkpoint_file = os.path.join(self.data_dir, "preparation_state.json")
        self.state = self.load_state()
        self.enc = tiktoken.get_encoding("gpt2")
        self._shutdown_requested = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        if self._shutdown_requested:
            # If shutdown already requested, force exit
            print(f"\nForce exit requested.")
            os._exit(1)
        
        self._shutdown_requested = True
        print(f"\nReceived signal {signum}. Saving state and exiting gracefully...")
        try:
            self.save_state()
        except Exception as e:
            print(f"Error saving state: {e}")
        finally:
            os._exit(0)
    
    def load_state(self):
        """Load preparation state from checkpoint file"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                print(f"Loaded state from checkpoint: {state}")
                return state
        return {
            "step": "download",
            "dataset_downloaded": False,
            "splits_created": False,
            "tokenization_complete": False,
            "binary_files": {}
        }
    
    def save_state(self):
        """Save current preparation state"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.state, f, indent=2)
        print(f"State saved to {self.checkpoint_file}")
    
    @retry_on_failure(max_retries=5, delay=10)
    def download_dataset(self):
        """Download the OpenWebText dataset with retry logic"""
        if self.state["dataset_downloaded"]:
            print("Dataset already downloaded, skipping...")
            return True
        
        print("Downloading OpenWebText dataset...")
        print("Note: This will download ~54GB to your HuggingFace cache directory")
        
        try:
            # Download with reduced number of processes to be more stable
            self.dataset = load_dataset(
                "Skylion007/openwebtext", 
                num_proc=4,  # Reduced from 8 for stability
                cache_dir=os.path.join(self.data_dir, ".cache")
            )
            
            self.state["dataset_downloaded"] = True
            self.state["step"] = "split"
            self.save_state()
            print("Dataset downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"Download failed: {e}")
            raise
    
    def create_splits(self):
        """Create train/validation splits"""
        if self.state["splits_created"] and hasattr(self, 'split_dataset'):
            print("Splits already created and loaded, skipping...")
            return
        
        print("Creating train/validation splits...")
        self.split_dataset = self.dataset["train"].train_test_split(
            test_size=0.0005, 
            seed=2357, 
            shuffle=True
        )
        self.split_dataset['val'] = self.split_dataset.pop('test')
        
        self.state["splits_created"] = True
        self.state["step"] = "tokenize"
        self.save_state()
        
        print(f"Splits created:")
        print(f"  Train: {len(self.split_dataset['train'])} samples")
        print(f"  Val: {len(self.split_dataset['val'])} samples")
    
    def process_example(self, example):
        """Process a single example for tokenization"""
        ids = self.enc.encode_ordinary(example['text'])
        ids.append(self.enc.eot_token)
        return {'ids': ids, 'len': len(ids)}
    
    @retry_on_failure(max_retries=3, delay=30)
    def tokenize_dataset(self):
        """Tokenize the dataset with retry logic"""
        if self.state["tokenization_complete"] and hasattr(self, 'tokenized'):
            print("Tokenization already complete and loaded, skipping...")
            return True
        
        print("Tokenizing dataset...")
        try:
            # Check if shutdown was requested
            if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
                return False
            
            # Try to load from cache first if tokenization was completed before
            if self.state["tokenization_complete"]:
                print("Attempting to load tokenized data from cache...")
                try:
                    self.tokenized = self.split_dataset.map(
                        self.process_example,
                        remove_columns=['text'],
                        desc="Loading tokenized data from cache",
                        num_proc=1,  # Single process to avoid conflicts
                        load_from_cache_file=True
                    )
                    print("Successfully loaded tokenized data from cache!")
                    return True
                except Exception as e:
                    print(f"Failed to load from cache: {e}")
                    print("Re-tokenizing dataset...")
                    self.state["tokenization_complete"] = False  # Reset to re-tokenize
            
            self.tokenized = self.split_dataset.map(
                self.process_example,
                remove_columns=['text'],
                desc="Tokenizing splits",
                num_proc=2,  # Reduced for stability (was 4)
                batch_size=1000  # Process in smaller batches
            )
            
            # Check again if shutdown was requested during tokenization
            if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
                print("Shutdown requested during tokenization, saving progress...")
                return False
            
            self.state["tokenization_complete"] = True
            self.state["step"] = "write_binary"
            self.save_state()
            print("Tokenization completed successfully!")
            return True
            
        except Exception as e:
            print(f"Tokenization failed: {e}")
            raise
    
    def write_binary_file(self, split_name, dataset):
        """Write a single binary file with progress tracking"""
        filename = os.path.join(self.data_dir, f'{split_name}.bin')
        
        # Check if this split is already complete
        if self.state["binary_files"].get(split_name, {}).get("complete", False):
            print(f"Binary file for {split_name} already complete, skipping...")
            return
        
        # Check if file already exists and ask user
        if os.path.exists(filename):
            existing_size = os.path.getsize(filename)
            if existing_size > 1000000:  # If file is larger than 1MB
                print(f"{filename} already exists ({existing_size/1024/1024:.1f} MB)")
                # In automated mode, assume we want to continue from checkpoint
                print("Checking if we can resume from existing progress...")
                resume_batch = self.state["binary_files"].get(split_name, {}).get("batch", 0)
                if resume_batch > 0:
                    print(f"Found resume point at batch {resume_batch}")
                else:
                    print("No valid resume point found, will recreate file")
                    os.remove(filename)
            else:
                # Small file, probably incomplete
                os.remove(filename)
        
        print(f"Writing binary file for {split_name}...")
        
        # Calculate total length and show token count
        arr_len = np.sum(dataset['len'], dtype=np.uint64)
        dtype = np.uint16
        print(f"Writing {arr_len:,} tokens to {filename}")
        
        # Create memory-mapped file if it doesn't exist
        if not os.path.exists(filename):
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        else:
            arr = np.memmap(filename, dtype=dtype, mode='r+', shape=(arr_len,))
        
        total_batches = 1024
        
        # Get resume point
        resume_batch = self.state["binary_files"].get(split_name, {}).get("batch", 0)
        
        try:
            idx = 0
            
            # If resuming, calculate starting index
            if resume_batch > 0:
                print(f"Resuming from batch {resume_batch}")
                for batch_idx in range(resume_batch):
                    batch = dataset.shard(
                        num_shards=total_batches, 
                        index=batch_idx, 
                        contiguous=True
                    ).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    idx += len(arr_batch)
                print(f"Skipped {idx:,} tokens (resuming from batch {resume_batch})")
            
            # Continue writing from resume point
            for batch_idx in tqdm(
                range(resume_batch, total_batches), 
                desc=f'Writing {filename}',
                initial=resume_batch,
                total=total_batches
            ):
                try:
                    batch = dataset.shard(
                        num_shards=total_batches, 
                        index=batch_idx, 
                        contiguous=True
                    ).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    
                    # Write to memory-mapped array
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                    
                    # Save progress every 50 batches
                    if batch_idx % 50 == 0:
                        if split_name not in self.state["binary_files"]:
                            self.state["binary_files"][split_name] = {}
                        self.state["binary_files"][split_name]["batch"] = batch_idx + 1
                        self.save_state()
                
                except Exception as e:
                    print(f"Error at batch {batch_idx}: {e}")
                    print(f"Written {idx:,} tokens so far")
                    # Save current progress
                    if split_name not in self.state["binary_files"]:
                        self.state["binary_files"][split_name] = {}
                    self.state["binary_files"][split_name]["batch"] = batch_idx
                    self.save_state()
                    raise
            
            # Flush and mark as complete
            arr.flush()
            
            if split_name not in self.state["binary_files"]:
                self.state["binary_files"][split_name] = {}
            self.state["binary_files"][split_name]["complete"] = True
            self.state["binary_files"][split_name]["batch"] = total_batches
            self.save_state()
            
            # Show completion info with file size
            file_size_gb = os.path.getsize(filename) / (1024**3)
            print(f"Successfully wrote {filename} ({file_size_gb:.1f} GB, {arr_len:,} tokens)")
            
        except Exception as e:
            print(f"Error writing {filename}: {e}")
            # Save current progress
            if split_name not in self.state["binary_files"]:
                self.state["binary_files"][split_name] = {}
            self.state["binary_files"][split_name]["batch"] = batch_idx if 'batch_idx' in locals() else 0
            self.save_state()
            raise
    
    def write_binary_files(self):
        """Write all binary files"""
        print("Writing binary files...")
        
        # Ensure tokenized dataset exists
        if not hasattr(self, 'tokenized'):
            print("Error: Tokenized dataset not found. This should not happen.")
            return
        
        for split_name, dataset in self.tokenized.items():
            self.write_binary_file(split_name, dataset)
        
        self.state["step"] = "complete"
        self.save_state()
    
    def cleanup(self):
        """Clean up checkpoint files after successful completion"""
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        print("Cleanup completed!")
    
    def run(self):
        """Run the complete preparation process"""
        print("Starting OpenWebText dataset preparation...")
        print("This script supports resuming from interruptions.")
        print("Press Ctrl+C to gracefully stop and save progress.\n")
        
        try:
            # Step 1: Download dataset
            if self.state["step"] in ["download"]:
                if not self.download_dataset():
                    return 1
            
            # Check for shutdown request
            if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
                return 0
            
            # Load dataset if we're resuming from later step
            if not hasattr(self, 'dataset'):
                print("Loading dataset from cache...")
                self.dataset = load_dataset(
                    "Skylion007/openwebtext", 
                    cache_dir=os.path.join(self.data_dir, ".cache")
                )
            
            # Step 2: Create splits (always run this to ensure split_dataset exists)
            if self.state["step"] in ["download", "split", "tokenize", "write_binary"]:
                self.create_splits()
                
            # Check for shutdown request
            if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
                return 0
            
            # Step 3: Tokenize
            if self.state["step"] in ["download", "split", "tokenize", "write_binary"]:
                if not self.tokenize_dataset():
                    print("Tokenization was interrupted. You can resume by running this script again.")
                    return 0
                    
            # Check for shutdown request
            if hasattr(self, '_shutdown_requested') and self._shutdown_requested:
                return 0
            
            # Step 4: Write binary files
            if self.state["step"] in ["download", "split", "tokenize", "write_binary"]:
                self.write_binary_files()
            
            print("\n🎉 Dataset preparation completed successfully!")
            
            # Show file sizes
            for split in ["train", "val"]:
                filepath = os.path.join(self.data_dir, f"{split}.bin")
                if os.path.exists(filepath):
                    size_gb = os.path.getsize(filepath) / (1024**3)
                    print(f"  {split}.bin: {size_gb:.1f} GB")
            
            self.cleanup()
            
        except KeyboardInterrupt:
            print("\nProcess interrupted by user.")
            self.save_state()
            print("You can resume by running this script again.")
            return 0
        except SystemExit:
            print("\nProcess terminated by system.")
            return 0
        except Exception as e:
            print(f"\nError: {e}")
            self.save_state()
            print("You can resume by running this script again.")
            return 1
        
        return 0

def main():
    """Main function"""
    preparer = DatasetPreparer()
    return preparer.run()

if __name__ == "__main__":
    exit(main())
