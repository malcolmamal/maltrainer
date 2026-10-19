import sys

import safetensors.torch as safetensors
import torch


def check_weights(path):
    try:
        tensors = safetensors.load_file(path)
        print(f"Checking statistics for {path}...")

        up_weights = [v for k, v in tensors.items() if "up" in k or "lora_B" in k]
        down_weights = [v for k, v in tensors.items() if "down" in k or "lora_A" in k]

        if not up_weights:
            print("No UP/B weights found!")
            return

        # Stack to check mean/std
        all_up = torch.cat([w.flatten() for w in up_weights])
        all_down = torch.cat([w.flatten() for w in down_weights])

        print(
            f"UP/B Weights - Mean: {all_up.mean().item():.6f}, Std: {all_up.std().item():.6f}, "
            f"Max: {all_up.max().item():.6f}"
        )
        print(
            f"DOWN/A Weights - Mean: {all_down.mean().item():.6f}, Std: {all_down.std().item():.6f}, "
            f"Max: {all_down.max().item():.6f}"
        )

        # Check for zeros
        if all_up.count_nonzero() == 0:
            print("WARNING: ALL UP/B WEIGHTS ARE ZERO!")
        else:
            print(f"Non-zero elements in UP/B: {all_up.count_nonzero().item()} / {all_up.numel()}")

    except Exception as e:
        print(f"Error reading {path}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_weights.py <path_to_lora>")
    else:
        check_weights(sys.argv[1])
