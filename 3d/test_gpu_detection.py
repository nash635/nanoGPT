#!/usr/bin/env python3
"""
GPU Detection Test for MFU Calculation
Tests the improved MFU calculation with proper H20/H100/A100 detection
"""

import torch

def test_gpu_detection():
    """Test GPU detection and peak FLOPS calculation"""
    try:
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return
            
        device_count = torch.cuda.device_count()
        print(f"🖥️  Found {device_count} GPU(s)")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_name_lower = gpu_name.lower()
            
            print(f"\n📋 GPU {i}: {gpu_name}")
            
            # Same logic as in the MFU calculation (simplified GPU list)
            if 'h20' in gpu_name_lower:
                peak_flops = 148e12  # Corrected: 296 TFLOPS is for FP8
                gpu_type = "H20"
            elif 'h100' in gpu_name_lower:
                peak_flops = 989e12 if 'sxm' in gpu_name_lower else 756e12
                gpu_type = "H100"
            elif 'a100' in gpu_name_lower:
                peak_flops = 312e12
                gpu_type = "A100"
            elif 'v100' in gpu_name_lower:
                peak_flops = 125e12
                gpu_type = "V100"
            elif 'p100' in gpu_name_lower:
                peak_flops = 18.7e12
                gpu_type = "P100"
            elif 'tesla' in gpu_name_lower:
                if 'v100' in gpu_name_lower:
                    peak_flops = 125e12
                    gpu_type = "Tesla V100"
                else:
                    peak_flops = 50e12
                    gpu_type = "Tesla (other)"
            else:
                peak_flops = 100e12
                gpu_type = "Unknown"
            
            print(f"   Type: {gpu_type}")
            print(f"   Peak FLOPS (bfloat16): {peak_flops/1e12:.0f} TFLOPS")
            
            # Get memory info
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"   Memory: {memory_gb:.1f} GB")
            print(f"   Compute Capability: {props.major}.{props.minor}")
            
        print(f"\n✅ GPU detection test completed")
        print(f"🔧 MFU calculation will use appropriate peak FLOPS for each GPU type")
        
    except Exception as e:
        print(f"❌ Error during GPU detection: {e}")

if __name__ == "__main__":
    test_gpu_detection()
