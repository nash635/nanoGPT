#!/usr/bin/env python3
"""
OpenWebText Dataset Cache Cleanup Script

This script helps you clean up various caches and intermediate files 
created during the OpenWebText dataset preparation process.
"""

import os
import shutil
import sys
from pathlib import Path

def get_size_str(size_bytes):
    """Convert bytes to human readable format"""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

def get_directory_size(path):
    """Calculate total size of directory"""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total

def cleanup_local_cache():
    """Clean up local .cache directory"""
    cache_dir = ".cache"
    if os.path.exists(cache_dir):
        size = get_directory_size(cache_dir)
        print(f"Found local cache directory: {cache_dir} ({get_size_str(size)})")
        
        response = input("Delete local .cache directory? (y/N): ").lower().strip()
        if response == 'y':
            try:
                shutil.rmtree(cache_dir)
                print(f"✅ Deleted {cache_dir}")
                return size
            except Exception as e:
                print(f"❌ Error deleting {cache_dir}: {e}")
                return 0
        else:
            print("Skipped local cache cleanup")
            return 0
    else:
        print("No local .cache directory found")
        return 0

def cleanup_huggingface_cache():
    """Clean up HuggingFace cache directory"""
    # Common HuggingFace cache locations
    hf_cache_locations = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/huggingface/datasets"),
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    
    total_freed = 0
    
    for cache_path in hf_cache_locations:
        if os.path.exists(cache_path):
            size = get_directory_size(cache_path)
            if size > 0:
                print(f"\nFound HuggingFace cache: {cache_path} ({get_size_str(size)})")
                
                # Show what's inside
                try:
                    items = os.listdir(cache_path)
                    if items:
                        print("Contents:")
                        for item in items[:10]:  # Show first 10 items
                            item_path = os.path.join(cache_path, item)
                            if os.path.isdir(item_path):
                                item_size = get_directory_size(item_path)
                                print(f"  📁 {item} ({get_size_str(item_size)})")
                            else:
                                try:
                                    item_size = os.path.getsize(item_path)
                                    print(f"  📄 {item} ({get_size_str(item_size)})")
                                except:
                                    print(f"  📄 {item}")
                        if len(items) > 10:
                            print(f"  ... and {len(items) - 10} more items")
                except:
                    pass
                
                response = input(f"Delete {cache_path}? (y/N): ").lower().strip()
                if response == 'y':
                    try:
                        shutil.rmtree(cache_path)
                        print(f"✅ Deleted {cache_path}")
                        total_freed += size
                    except Exception as e:
                        print(f"❌ Error deleting {cache_path}: {e}")
                else:
                    print(f"Skipped {cache_path}")
    
    if total_freed == 0:
        print("No HuggingFace cache directories found or none deleted")
    
    return total_freed

def cleanup_checkpoint_files():
    """Clean up checkpoint and state files"""
    checkpoint_files = [
        "preparation_state.json",
        "dataset_checkpoint.pkl",
        "tokenized_checkpoint.pkl",
        "progress_checkpoint.json"
    ]
    
    total_freed = 0
    found_files = []
    
    for filename in checkpoint_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            found_files.append((filename, size))
            total_freed += size
    
    if found_files:
        print(f"\nFound checkpoint files:")
        for filename, size in found_files:
            print(f"  📄 {filename} ({get_size_str(size)})")
        
        response = input("Delete checkpoint files? (y/N): ").lower().strip()
        if response == 'y':
            deleted_size = 0
            for filename, size in found_files:
                try:
                    os.remove(filename)
                    print(f"✅ Deleted {filename}")
                    deleted_size += size
                except Exception as e:
                    print(f"❌ Error deleting {filename}: {e}")
            return deleted_size
        else:
            print("Skipped checkpoint files")
            return 0
    else:
        print("No checkpoint files found")
        return 0

def cleanup_binary_files():
    """Clean up generated binary files"""
    binary_files = ["train.bin", "val.bin"]
    total_size = 0
    found_files = []
    
    for filename in binary_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            found_files.append((filename, size))
            total_size += size
    
    if found_files:
        print(f"\nFound binary data files:")
        for filename, size in found_files:
            print(f"  📄 {filename} ({get_size_str(size)})")
        print(f"Total size: {get_size_str(total_size)}")
        
        response = input("Delete binary files? (y/N): ").lower().strip()
        if response == 'y':
            deleted_size = 0
            for filename, size in found_files:
                try:
                    os.remove(filename)
                    print(f"✅ Deleted {filename}")
                    deleted_size += size
                except Exception as e:
                    print(f"❌ Error deleting {filename}: {e}")
            return deleted_size
        else:
            print("Skipped binary files")
            return 0
    else:
        print("No binary files found")
        return 0

def cleanup_temporary_files():
    """Clean up temporary and log files"""
    temp_patterns = [
        "*.tmp",
        "*.log",
        "*.pyc",
        "__pycache__",
        ".DS_Store"
    ]
    
    import glob
    found_files = []
    total_size = 0
    
    for pattern in temp_patterns:
        if pattern == "__pycache__":
            # Handle __pycache__ directories
            for pycache_dir in glob.glob("**/__pycache__", recursive=True):
                if os.path.isdir(pycache_dir):
                    size = get_directory_size(pycache_dir)
                    found_files.append((pycache_dir, size, "dir"))
                    total_size += size
        else:
            # Handle file patterns
            for filepath in glob.glob(pattern):
                if os.path.isfile(filepath):
                    size = os.path.getsize(filepath)
                    found_files.append((filepath, size, "file"))
                    total_size += size
    
    if found_files:
        print(f"\nFound temporary files:")
        for filepath, size, file_type in found_files:
            icon = "📁" if file_type == "dir" else "📄"
            print(f"  {icon} {filepath} ({get_size_str(size)})")
        
        response = input("Delete temporary files? (y/N): ").lower().strip()
        if response == 'y':
            deleted_size = 0
            for filepath, size, file_type in found_files:
                try:
                    if file_type == "dir":
                        shutil.rmtree(filepath)
                    else:
                        os.remove(filepath)
                    print(f"✅ Deleted {filepath}")
                    deleted_size += size
                except Exception as e:
                    print(f"❌ Error deleting {filepath}: {e}")
            return deleted_size
        else:
            print("Skipped temporary files")
            return 0
    else:
        print("No temporary files found")
        return 0

def show_current_usage():
    """Show current disk usage in the directory"""
    current_dir = "."
    total_size = get_directory_size(current_dir)
    
    print(f"\nCurrent directory disk usage:")
    print(f"📁 {os.path.abspath(current_dir)}: {get_size_str(total_size)}")
    
    # Show largest files/directories
    items = []
    try:
        for item in os.listdir(current_dir):
            item_path = os.path.join(current_dir, item)
            if os.path.isdir(item_path):
                size = get_directory_size(item_path)
                items.append((item, size, "dir"))
            elif os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                items.append((item, size, "file"))
    except:
        pass
    
    # Sort by size and show top 10
    items.sort(key=lambda x: x[1], reverse=True)
    if items:
        print("\nLargest items:")
        for item, size, item_type in items[:10]:
            icon = "📁" if item_type == "dir" else "📄"
            print(f"  {icon} {item}: {get_size_str(size)}")

def main():
    """Main cleanup function"""
    print("🧹 OpenWebText Dataset Cache Cleanup")
    print("=" * 50)
    
    # Show current usage first
    show_current_usage()
    
    print("\n" + "=" * 50)
    print("Cleanup Options:")
    print("=" * 50)
    
    total_freed = 0
    
    # 1. Local cache
    print("\n1. Local Cache (.cache directory)")
    print("-" * 30)
    total_freed += cleanup_local_cache()
    
    # 2. HuggingFace cache
    print("\n2. HuggingFace Cache")
    print("-" * 30)
    total_freed += cleanup_huggingface_cache()
    
    # 3. Checkpoint files
    print("\n3. Checkpoint Files")
    print("-" * 30)
    total_freed += cleanup_checkpoint_files()
    
    # 4. Binary files
    print("\n4. Generated Binary Files")
    print("-" * 30)
    total_freed += cleanup_binary_files()
    
    # 5. Temporary files
    print("\n5. Temporary Files")
    print("-" * 30)
    total_freed += cleanup_temporary_files()
    
    # Summary
    print("\n" + "=" * 50)
    print("Cleanup Summary")
    print("=" * 50)
    if total_freed > 0:
        print(f"✅ Total space freed: {get_size_str(total_freed)}")
    else:
        print("ℹ️  No files were deleted")
    
    print("\nCleanup completed!")

def interactive_menu():
    """Interactive menu for selective cleanup"""
    while True:
        print("\n🧹 OpenWebText Dataset Cache Cleanup")
        print("=" * 50)
        print("1. Show current disk usage")
        print("2. Clean local .cache directory")
        print("3. Clean HuggingFace cache")
        print("4. Clean checkpoint files")
        print("5. Clean binary files (train.bin, val.bin)")
        print("6. Clean temporary files")
        print("7. Clean everything (interactive)")
        print("8. Exit")
        print("=" * 50)
        
        choice = input("Select option (1-8): ").strip()
        
        if choice == '1':
            show_current_usage()
        elif choice == '2':
            cleanup_local_cache()
        elif choice == '3':
            cleanup_huggingface_cache()
        elif choice == '4':
            cleanup_checkpoint_files()
        elif choice == '5':
            cleanup_binary_files()
        elif choice == '6':
            cleanup_temporary_files()
        elif choice == '7':
            main()
            break
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # Non-interactive mode - clean everything
            main()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python cleanup_cache.py           # Interactive mode")
            print("  python cleanup_cache.py --all     # Clean everything (interactive)")
            print("  python cleanup_cache.py --help    # Show this help")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Interactive mode
        interactive_menu()
