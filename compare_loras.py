import sys
from pathlib import Path

import safetensors.torch as safetensors
from safetensors import SafetensorError


def analyze_lora(path):
    if not Path(path).exists():
        print(f"File not found: {path}")
        return None

    try:
        tensors = safetensors.load_file(path)
    except (OSError, ValueError, RuntimeError, KeyError, SafetensorError) as e:
        print(f"Error loading {path}: {e}")
        return None

    total_params = 0
    modules = set()
    ranks = set()

    for k, v in tensors.items():
        total_params += v.numel()

        # simple parsing to find module name
        # typical format: prefix.module.lora_A.weight
        parts = k.split(".")
        if "lora_" in k:
            # find where lora_A or lora_B or lora_down etc is
            # usually at the end
            pass

        # Just remove lora_A/B/weight to get module path
        clean_k = (
            k.replace(".lora_A.weight", "")
            .replace(".lora_B.weight", "")
            .replace(".lora_down.weight", "")
            .replace(".lora_up.weight", "")
            .replace(".alpha", "")
        )

        # Try to identify the core module name (e.g. attention.to_q)
        # Z-Image keys: diffusion_model.layers.0.attention.to_q

        # Heuristic to group
        if "layers" in clean_k:
            # extract "layers.X.SUBMODULE"
            try:
                # find index of 'layers'
                idx = parts.index("layers")
                # get submodule (rest)
                submodule = (
                    ".".join(parts[idx + 2 :])
                    .replace(".lora_A.weight", "")
                    .replace(".lora_B.weight", "")
                    .replace(".alpha", "")
                )
                modules.add(submodule)
            except ValueError:
                modules.add(clean_k)
        else:
            modules.add(clean_k)

        # Check rank from lora_A/down (input -> rank) or lora_B/up (rank -> output)
        if "lora_A" in k or "lora_down" in k:
            ranks.add(v.shape[0])  # usually weight is [rank, in_dim] or linear is [rank, in]??
            # torch.nn.Linear weight is [out, in].
            # lora_down is In -> Rank. Weight is [Rank, In].
            # lora_up is Rank -> Out. Weight is [Out, Rank].
            ranks.add(v.shape[0])

    return {
        "size_mb": Path(path).stat().st_size / (1024 * 1024),
        "params": total_params,
        "modules": sorted(modules),
        "ranks": sorted(ranks),
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_loras.py <good_lora> <my_lora>")
        return

    good_path = sys.argv[1]
    my_path = sys.argv[2]

    print(f"--- Analyzing Good LoRA: {Path(good_path).name} ---")
    good_stats = analyze_lora(good_path)
    if good_stats:
        print(f"Size: {good_stats['size_mb']:.2f} MB")
        print(f"Total Params: {good_stats['params']:,}")
        print(f"Ranks detected: {good_stats['ranks']}")
        print(f"Target Modules: {good_stats['modules']}")

    print(f"\n--- Analyzing My LoRA: {Path(my_path).name} ---")
    my_stats = analyze_lora(my_path)
    if my_stats:
        print(f"Size: {my_stats['size_mb']:.2f} MB")
        print(f"Total Params: {my_stats['params']:,}")
        print(f"Ranks detected: {my_stats['ranks']}")
        print(f"Target Modules: {my_stats['modules']}")

    if good_stats and my_stats:
        print("\n--- Comparison ---")
        good_mods = set(good_stats["modules"])
        my_mods = set(my_stats["modules"])

        missing = good_mods - my_mods
        extra = my_mods - good_mods

        if missing:
            print(f"MISSING modules in My LoRA: {missing}")
        else:
            print("My LoRA covers all modules in Good LoRA.")

        if extra:
            print(f"EXTRA modules in My LoRA: {extra}")


if __name__ == "__main__":
    main()
