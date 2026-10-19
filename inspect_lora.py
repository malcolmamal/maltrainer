import sys

import safetensors.torch as safetensors


def inspect_lora(path):
    try:
        tensors = safetensors.load_file(path)
        print(f"Total keys: {len(tensors)}")
        print("First 10 keys:")
        for k in list(tensors.keys())[:10]:
            print(k)
    except Exception as e:
        print(f"Error reading {path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_lora.py <path_to_lora>")
    else:
        inspect_lora(sys.argv[1])
