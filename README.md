# 🏜️ Offroad Autonomy Semantic Segmentation 
**Duality AI Hackathon - Winning Submission**

🤗 **[Hugging Face Model Weights: SegFormer-B2](https://huggingface.co/Yashashvi-0508/offroad-segmentation-segformer-b2)**

## 📌 Overview
This repository contains our solution for the **Duality AI Offroad Autonomy Segmentation Challenge**. The objective is to train a robust semantic segmentation model using synthetic digital twin data (from FalconEditor) capable of segmenting 10 distinct off-road classes (Trees, Rocks, Logs, etc.) and generalizing to a novel, unseen desert environment.

By discarding the heavy baseline (DINOv2) and engineered a heavily optimized **SegFormer-B2** architecture, we successfully bridged the synthetic-to-synthetic domain gap and handled extreme class imbalances, doubling the baseline mIoU while running comfortably on consumer-grade hardware (6GB VRAM) at < 50ms inference time.

---

## 🚀 Key Innovations & Methodology

1. **Defeating the Domain Shift (`dataset.py`)**
   Naive models overfit to the training map's specific lighting and shadows. We replaced standard transforms with an aggressive `Albumentations` pipeline. We used `RandomShadow` to simulate dynamic time-of-day lighting variations and `CoarseDropout` to simulate physical occlusion, forcing the network to identify rare obstacles even when partially covered by ground clutter.
2. **Solving Catastrophic Class Imbalance (`losses.py` & `utils.py`)**
   Our EDA revealed that Sky, Landscape, and Dry Grass made up **81%** of the dataset, while critical obstacles like **Logs** comprised only **0.078%**. We implemented a custom **Frequency-Weighted CrossEntropy + Dice Loss**. By assigning a massive static weight multiplier (8.15x) to the Logs class, we forced the network to penalize majority classes and learn critical navigational hazards.
3. **Hardware-Efficient Architecture (`train_segformer.py`)**
   We transitioned to `SegformerForSemanticSegmentation` (mit-b2) paired with PyTorch Mixed Precision (`torch.amp.autocast`). This halved VRAM consumption, allowing us to train locally with a batch size of 4 at 512x512 resolution without OOM errors.

---

## 📊 Results

| Model | Setup | Final mIoU | Inference Speed |
| :--- | :--- | :--- | :--- |
| Baseline (DINOv2 vits14) | Base CE Loss, No Augments | 0.2951 | ~ 75ms |
| **Ours (SegFormer-B2)** | **Weighted CE + Dice, Albumentations** | **0.5506** | **< 45ms** |

---

## ⚙️ Environment & Dependency Requirements

This project was developed and trained locally on a Windows 11 machine with an NVIDIA RTX 4050 (6GB VRAM) using CUDA 12.x. 

**Required Packages:**
* `torch` & `torchvision` (PyTorch 2.x+)
* `transformers` (HuggingFace)
* `albumentations` (High-performance image augmentations)
* `opencv-python-headless` (Fast image I/O without GUI dependencies)
* `numpy`, `tqdm`, `matplotlib`

**Setup Instructions:**
1. Clone this repository:
   ```bash
   git clone [https://github.com/Yashashvi-05/offroad-segmentation.git](https://github.com/Yashashvi-05/offroad-segmentation.git)
   cd offroad-segmentation
