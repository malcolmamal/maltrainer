# Z-Image LoRA: Comparative Analysis & Quality Optimization Guide

This report summarizes the technical differences between **maltrainer** and other industry-standard trainers (AI-Toolkit, Kohya_ss, OneTrainer, SimpleTuner, Musubi) and outlines how to maximize output quality on high-end hardware like the **RTX 5090**.

## 1. Technical Comparison: Trainer Architectures

| Feature | **maltrainer** (Ours) | **AI-Toolkit** | **SimpleTuner** | **Musubi-Tuner** | **Kohya_ss** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logic Unit** | Step-Centric (Global) | Step-Centric | Bucket-Centric | DiT-Specialized | Folder-Repeats |
| **Philosophy** | Code Clarity | Production Env | Scaling/Flexibility | Mathematical DiT | Legacy/Standard |
| **Z-Image Support**| Native | Native (Extension)| Deep/Multimodal | Dedicated | Experimental |
| **Result Size** | 156MB (Config) | 170MB - 300MB | 200MB - 500MB | Highly Optimized | Variable |

## 2. Advanced Competitors: SimpleTuner & Musubi

### SimpleTuner (The "Swiss Army Knife")
SimpleTuner is the preferred tool for high-end researchers and professional trainers.
*   **Strengths**: Best-in-class **Aspect Ratio Bucketing** and **Multi-GPU** scaling. It can handle datasets where images have wildly different dimensions without cropping them into squares.
*   **Why it's different**: It uses extreme caching (text embeddings and VAE latents are pre-calculated and stored on disk) to allow training massive models on relatively consumer hardware.
*   **RTX 5090 Advantage**: With 32GB VRAM, you wouldn't need as much caching, making SimpleTuner incredibly fast for you.

### Musubi-Tuner (The "DiT Specialist")
Musubi is focused almost exclusively on Diffusion Transformer (DiT) models like Z-Image, Flux, and SD3.
*   **Strengths**: Deep integration of **Flow Matching** math. It often produces "smoother" gradients and fewer artifacts than more general trainers.
*   **Why it's different**: It implements advanced sampling techniques (like Log-Normal sampling) directly into the training loop, which ensures that the model learns the "noisy" and "detail" parts of the image with equal importance.

## 3. The "170MB" Quality Threshold

During this session, we increased our output from **8MB** to **156MB**. Here is the architectural reason for that shift:

*   **Small (8MB):** Targets only the **Attention** layers (`to_q`, `to_k`, `to_v`). This captures the *composition* of the subject but often lacks the fine *textures* or unique *lighting* of the dataset.
*   **Large (156MB - 170MB):** Targets both **Attention** and **MLP/Feed-Forward** layers (`w1`, `w2`, `w3`). The MLP layers act as the "memory banks" for what things look like. By adding these, the model can actually "memorize" the fine details of a specific face or style.

## 3. High-Tier Quality Optimization Strategies

To move from "good" to "photorealistic/perfect" LoRAs, consider these improvements:

### A. Optimizer Evolution (Blackwell Optimization)
Currently using **AdamW**. While stable, it requires a static learning rate.
*   **Improvement**: Use **Prodigy** or **D-Adaptation**. These are "Adaptive" optimizers that automatically find the best learning rate for your specific dataset on the fly. 
*   **Blackwell Benefit**: Since the 5090 handles high precision so well, these optimizers converge much faster (saving 30-40% training time).

### B. Aspect Ratio Bucketing
Currently, the trainer uses a fixed resolution (512x512).
*   **The Issue**: If your dataset has portraits (9:16) or landscapes (16:9), the model is only learning a "cropped" version.
*   **Improvement**: Implement **Bucketing**. The trainer groups images of similar ratios (e.g., all 448x640) and trains them together. This prevents the model from developing a "square bias" and improves anatomical accuracy.

### C. Captioning Strategy (Natural Language vs. Tags)
*   **Kohya-style (Tags)**: `sks woman, blonde hair, blue eyes, cinematic lighting`
*   **AI-Toolkit style (Natural Language)**: `A cinematic photo of sks woman with blonde hair and striking blue eyes, looking at the camera.`
*   **Z-Image Recommendation**: Z-Image uses the **Qwen** text encoder. Qwen is a Large Language Model (LLM). It prefers **Natural Language**. Use one sentence that describes the whole image rather than a list of tags.

### D. Learning Rate Scheduling
*   **Standard**: Constant or Linear decay.
*   **Advanced**: **Cosine with Restarts**. This "shakes" the model out of local minima (bad guesses) by spiking the learning rate occasionally before cooling it down again. This often results in much sharper details.

## 4. Summary of Improvements Made in this Session

1.  **Architecture**: Shifted from Attention-only to **Full Block (Attn + MLP)** coverage.
2.  **Compatibility**: Standardized naming for **ComfyUI** (dots vs underscores).
3.  **Stability**: Removed Windows-multiprocessing bottlenecks (VAE offloading).
4.  **Hardware Utilization**: Leveraged **BF16** and **Gradient Checkpointing** to fit 14B parameters of weights into 32GB VRAM.
5.  **UX**: Implemented a **Global-Step** progress bar with epoch-postfix.

## 5. Next Steps for Peak Quality
If the 156MB LoRA is still not sharp enough:
1.  **Increase resolution** to 1024x1024 (The 5090 has the VRAM for it).
2.  **Switch to Natural Language captions** using the `Qwen-VL` captioner tool.
3.  **Lower the training LR** to `5e-5` and double the epochs for a "slow cook" approach.
