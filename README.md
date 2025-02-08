# MediVision

## Project Overview
This project focuses on enhancing low-resolution medical images using Super-Resolution Generative Adversarial Networks (SRGANs). In remote or resource-limited locations, access to high-quality medical imaging equipment is often restricted, making diagnosis difficult. This project aims to provide a solution by using SRGANs to upscale low-resolution medical images, ensuring that crucial diagnostic details are preserved.

## Objective
- Improve the resolution of low-quality medical images, particularly MRI scans.
- Assist physicians in making accurate diagnoses by restoring details in low-resolution images.
- Provide an alternative to expensive imaging equipment for underfunded or remote medical facilities.

## Dataset
- A total of **5,712 MRI brain images** were used.
- Images were divided into:
  - **Low-resolution versions**: 32x32 pixels
  - **High-resolution versions**: 128x128 pixels (used as ground truth)
- The SRGAN model was trained to enhance low-resolution images to match high-resolution ground truth.

## Challenges
- Low-resolution medical images obscure critical diagnostic features such as small tumors or lesions.
- Conventional upscaling methods often produce blurry or distorted images, making them unreliable for medical diagnosis.
- Remote areas with limited access to advanced imaging technology need a cost-effective alternative.

## Solution: SRGAN Model
- The **SRGAN model** was used to enhance image resolution.
- The **CBAM (Convolutional Block Attention Module)** was incorporated to improve feature extraction.
- **Perceptual loss** using VGG19 features was implemented to maintain high fidelity in reconstructed images.
- The model was evaluated using **SSIM (Structural Similarity Index Measure)** and **PSNR (Peak Signal-to-Noise Ratio)**.

## Results & Conclusion
- The model successfully enhanced image quality, making it feasible for deployment in resource-limited settings.
- High **SSIM and PSNR scores** confirmed the effectiveness of the method.
- Testing demonstrated reliable high-resolution outputs across varied conditions.
- The project contributes to **improving diagnostic accuracy in remote healthcare facilities**.

