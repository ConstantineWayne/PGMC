# Pre-trained Weights and Datasets

Please download the pretrained weights of the four multimodal 3D encoders as well as the datasets from  
the official code of **“Point-Cache: Test-time Dynamic and Hierarchical Cache for Robust and Generalizable Point Cloud Analysis” (CVPR'25)**.

Place the corresponding weight paths into the appropriate functions in `utils/utils.py`,  
e.g., `load_ulip`.

Replace all occurrences of `path/to/data` in the configs in `utils/utils.py` with the actual dataset paths.

---

# Package Setup

The experiments were conducted under the following environment:
 
- **Python:** 3.8.16  
- **PyTorch:** 1.12.0  
- **CUDA:** 11.6  
- **torchvision:** 0.13.0  
- **timm:** 0.9.16  
- **Dassl:** (installed based on the official repository instructions)  
- **pueue & pueued:** 2.0.4  

Make sure all dependencies are installed before running the experiments.

---

# Usage

Taking **ULIP-1** as an example for obtaining results on **ModelNet-C** with *jitter* corruption, run:

> **Note:** Update the `noise_type` inside the `get_original_dataloader` function in  
> `utils/utils.py` to match the corruption type used during inference.

```bash
nohup bash ./scripts/main.sh 0 ulip ./pointbert_ulip1.pt modelnet_c obj_only jitter_2 1024 vitg14 ulip1 hierarchical so_obj_only_9 &
