#/src/utils/gpu_troubleshoot.py

"""
TensorFlow GPU Troubleshooting Script for Windows 10 WSL
This script diagnoses common issues preventing TensorFlow from detecting GPUs
"""

import os
import sys
import subprocess
import platform
import warnings
#warnings.filterwarnings('ignore')

def run_command(command):
    """Execute a shell command and return output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, 
                              text=True, timeout=30)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)

def check_system_info():
    """Check basic system information"""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    print(f"Platform: {platform.platform()}")
    print(f"Python Version: {sys.version}")
    
    # Check if running in WSL
    ret_code, output, _ = run_command("uname -r")
    if ret_code == 0 and "microsoft" in output.lower():
        print("✓ Running in WSL")
        
        # Check WSL version
        ret_code, output, _ = run_command("wsl.exe -l -v")
        if ret_code == 0:
            print(f"WSL Info: {output}")
    else:
        print("⚠ Not running in WSL or unable to detect")
    
    print()

def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("=" * 60)
    print("NVIDIA DRIVER CHECK")
    print("=" * 60)
    
    ret_code, output, error = run_command("nvidia-smi")
    if ret_code == 0:
        print("✓ NVIDIA driver detected")
        lines = output.split('\n')
        for line in lines:
            if "Driver Version" in line:
                print(f"Driver info: {line.strip()}")
                break
        
        # Show GPU info
        print("\nGPU Information:")
        for line in lines:
            if "Tesla\|GeForce\|Quadro\|RTX" in line or "MiB" in line:
                print(f"  {line.strip()}")
    else:
        print("✗ NVIDIA driver not found or nvidia-smi not available")
        print("Solutions:")
        print("  1. Install NVIDIA drivers on Windows host")
        print("  2. Ensure WSL2 is being used (not WSL1)")
        print("  3. Update Windows to support CUDA on WSL2")
    
    print()

def check_cuda_installation():
    """Check CUDA installation"""
    print("=" * 60)
    print("CUDA INSTALLATION CHECK")
    print("=" * 60)
    
    # Check nvcc
    ret_code, output, _ = run_command("nvcc --version")
    if ret_code == 0:
        print("✓ CUDA toolkit found")
        print(f"NVCC output: {output}")
    else:
        print("✗ CUDA toolkit not found")
        print("Install CUDA toolkit in WSL:")
        print("  wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb")
        print("  sudo dpkg -i cuda-keyring_1.0-1_all.deb")
        print("  sudo apt-get update")
        print("  sudo apt-get -y install cuda-toolkit")
    
    # Check CUDA environment variables
    cuda_home = os.environ.get('CUDA_HOME', os.environ.get('CUDA_PATH', ''))
    if cuda_home:
        print(f"✓ CUDA_HOME/CUDA_PATH: {cuda_home}")
    else:
        print("⚠ CUDA_HOME/CUDA_PATH not set")
    
    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'cuda' in ld_path.lower():
        print(f"✓ CUDA in LD_LIBRARY_PATH: {ld_path}")
    else:
        print("⚠ CUDA not in LD_LIBRARY_PATH")
        print("Add to ~/.bashrc:")
        print("  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")
    
    print()

def check_cudnn():
    """Check cuDNN installation"""
    print("=" * 60)
    print("cuDNN CHECK")
    print("=" * 60)
    
    # Check for cuDNN files
    cudnn_paths = [
        '/usr/local/cuda/include/cudnn.h',
        '/usr/include/cudnn.h',
        '/usr/local/cuda/lib64/libcudnn.so'
    ]
    
    found_cudnn = False
    for path in cudnn_paths:
        if os.path.exists(path):
            print(f"✓ Found cuDNN file: {path}")
            found_cudnn = True
    
    if not found_cudnn:
        print("✗ cuDNN not found")
        print("Install cuDNN:")
        print("  sudo apt-get install libcudnn8 libcudnn8-dev")
    
    print()

def check_tensorflow():
    """Check TensorFlow installation and GPU detection"""
    print("=" * 60)
    print("TENSORFLOW CHECK")
    print("=" * 60)
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow version: {tf.__version__}")
        
        # Check if TensorFlow is GPU-enabled
        if hasattr(tf.config, 'list_physical_devices'):
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"✓ TensorFlow detects {len(gpus)} GPU(s)")
                for i, gpu in enumerate(gpus):
                    print(f"  GPU {i}: {gpu}")
                
                # Test GPU availability
                print("\nTesting GPU computation...")
                try:
                    with tf.device('/GPU:0'):
                        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                        c = tf.matmul(a, b)
                        print(f"✓ GPU computation successful: {c.numpy()}")
                except Exception as e:
                    print(f"✗ GPU computation failed: {e}")
            else:
                print("✗ No GPUs detected by TensorFlow")
                print("\nTroubleshooting steps:")
                print("1. Install GPU-enabled TensorFlow:")
                print("   pip install tensorflow[and-cuda]")
                print("2. Or install specific versions:")
                print("   pip install tensorflow-gpu")
        
        # Check TensorFlow build info
        print(f"\nTensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"GPUs available: {gpus if gpus else None}")
        
    except ImportError:
        print("✗ TensorFlow not installed")
        print("Install TensorFlow with GPU support:")
        print("  pip install tensorflow[and-cuda]")
    except Exception as e:
        print(f"✗ Error checking TensorFlow: {e}")
    
    print()

def check_python_packages():
    """Check relevant Python packages"""
    print("=" * 60)
    print("PYTHON PACKAGES CHECK")
    print("=" * 60)
    
    packages = ['tensorflow', 'tensorflow-gpu', 'numpy', 'nvidia-ml-py3']
    
    for package in packages:
        ret_code, output, _ = run_command(f"pip show {package}")
        if ret_code == 0:
            version = [line for line in output.split('\n') if line.startswith('Version:')]
            if version:
                print(f"✓ {package}: {version[0].split(': ')[1]}")
        else:
            print(f"- {package}: Not installed")
    
    print()

def provide_solutions():
    """Provide common solutions"""
    print("=" * 60)
    print("COMMON SOLUTIONS")
    print("=" * 60)
    
    solutions = [
        "1. Ensure you're using WSL2 (not WSL1):",
        "   wsl --set-version <distribution> 2",
        "",
        "2. Install NVIDIA drivers on Windows host (not in WSL)",
        "",
        "3. Install CUDA toolkit in WSL:",
        "   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb",
        "   sudo dpkg -i cuda-keyring_1.0-1_all.deb",
        "   sudo apt-get update",
        "   sudo apt-get -y install cuda-toolkit-12-2",
        "",
        "4. Install cuDNN:",
        "   sudo apt-get install libcudnn8 libcudnn8-dev",
        "",
        "5. Install TensorFlow with GPU support:",
        "   pip install tensorflow[and-cuda]",
        "",
        "6. Set environment variables in ~/.bashrc:",
        "   export CUDA_HOME=/usr/local/cuda",
        "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "   export PATH=/usr/local/cuda/bin:$PATH",
        "",
        "7. Restart WSL after installations:",
        "   wsl --shutdown",
        "   wsl",
        "",
        "8. Check Windows version (requires Windows 10 21H2+ or Windows 11)"
    ]
    
    for solution in solutions:
        print(solution)
    
    print()

def main():
    """Main troubleshooting function"""
    print("TensorFlow GPU Troubleshooting Script for Windows 10 WSL")
    print("This script will check your system for common GPU detection issues.\n")
    
    check_system_info()
    check_nvidia_driver()
    check_cuda_installation()
    check_cudnn()
    check_python_packages()
    check_tensorflow()
    provide_solutions()
    
    print("Troubleshooting complete!")
    print("If issues persist, check the TensorFlow GPU guide:")
    print("https://www.tensorflow.org/install/gpu")

if __name__ == "__main__":
    main()
