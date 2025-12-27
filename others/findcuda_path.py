# find_cuda_path.py
import os
import glob

def find_cuda_path():
    # 常见安装路径
    possible_paths = [

        "f:/nvidia-12.6",
        os.path.expandvars("f:/nvidia-12.6")
    ]
    
    for base_path in possible_paths:
        if os.path.exists(base_path):
            versions = glob.glob(os.path.join(base_path, "v*"))
            if versions:
                print(f"Found CUDA at: {base_path}")
                for version in versions:
                    include_path = os.path.join(version, "include")
                    if os.path.exists(include_path):
                        print(f"  Version: {os.path.basename(version)}")
                        print(f"  Include path: {include_path}")
                        return include_path
    return None

cuda_path = find_cuda_path()
if cuda_path:
    print(f"\nUse this path in VS Code: {cuda_path}")
else:
    print("CUDA not found. Please check installation.")