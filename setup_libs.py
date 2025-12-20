import os
import shutil
import site
import sys
from distutils.dir_util import copy_tree

def get_package_path(package_name):
    import importlib.util
    spec = importlib.util.find_spec(package_name)
    if spec and spec.origin:
        return os.path.dirname(spec.origin)
    return None

def setup_local_lib():
    local_lib = "local_lib"
    if not os.path.exists(local_lib):
        os.makedirs(local_lib)

    # Handle CLIP
    try:
        import clip
        clip_path = os.path.dirname(clip.__file__)
        print(f"Found CLIP at: {clip_path}")
        target_clip = os.path.join(local_lib, "clip")
        if os.path.exists(target_clip):
            shutil.rmtree(target_clip)
        shutil.copytree(clip_path, target_clip)
        
        # Replace files
        print("Replacing CLIP files...")
        shutil.copy("replace/clip.py", os.path.join(target_clip, "clip.py"))
        shutil.copy("replace/model.py", os.path.join(target_clip, "model.py"))
        
    except ImportError:
        print("CLIP not found. Please install it first: pip install git+https://github.com/openai/CLIP.git")

    # Handle Torchvision
    try:
        import torchvision
        tv_path = os.path.dirname(torchvision.__file__)
        print(f"Found Torchvision at: {tv_path}")
        target_tv = os.path.join(local_lib, "torchvision")
        if os.path.exists(target_tv):
            shutil.rmtree(target_tv)
        shutil.copytree(tv_path, target_tv)
        
        # Replace files
        print("Replacing Torchvision files...")
        copy_tree("replace/torchvision.datasets", os.path.join(target_tv, "datasets"))
        
    except ImportError:
        print("Torchvision not found. Please install it.")

if __name__ == "__main__":
    setup_local_lib()
