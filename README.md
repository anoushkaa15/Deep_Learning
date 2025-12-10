# Deep_Learning

<img width="1136" height="483" alt="image" src="https://github.com/user-attachments/assets/71016d07-e129-4466-95d6-1c513667c148" />



# Anomaly Detection in Aerial Agricultural Images: A Comparative Study of Supervised and Self-Supervised Approaches
Krish Gupta, Garv Chadha, Kabir Gupta, Anoushka Yadav
December 2025

________________________________________

## Contents
1 Abstract  
2 Introduction  
2.1 Problem Statement  
2.2 Our Contributions  
2.3 System Pipeline Overview  
3 Related Work  
3.1 Foundational Approaches in Vision Transformers  
3.2 Reconstruction-Based Anomaly Detection  
3.3 Anomaly Suppression Mechanisms  
4 Methodology  
4.1 Phase 1: Conditional GAN (Supervised)  
4.2 Phase 2: CNN-Based Autoencoder (Unsupervised)  
4.3 Phase 3: Swin-MAE (Self-Supervised + Fine-Tuning)  
4.4 Phase 4: Label-Free Dynamic ASL (Proposed Method)  
5 Results  
5.1 Quantitative Results  
5.2 Qualitative Visual Analysis  
6 Discussion  
6.1 Why Traditional Methods Fail  
6.2 The Problem of Reconstruction Collapse  
6.3 Lessons from Experimentation  
7 Conclusion  
7.1 Summary  
7.2 Future Directions  
8 References  

________________________________________

## 1. Abstract
We present a comprehensive study on Anomaly Detection in Aerial Agricultural Images, aiming to solve the critical challenge of identifying crop stress, nutrient deficiency, and irregular growth patterns without relying on expensive pixel-level annotations. Unlike industrial datasets, agricultural imagery presents unique challenges including high intra-class variation, unstructured geometry, and significant natural noise.
This report documents our methodological evolution, starting from supervised approaches using Conditional GANs (cGAN), moving to basic CNN-based autoencoders, and finally adopting a state-of-the-art Swin Transformer Masked Autoencoder (Swin-MAE) framework. We demonstrate that while supervised methods fail due to class imbalance and label noise (mIoU 0.1547), and basic autoencoders suffer from "reconstruction collapse" (mIoU 0.039), our implementation of a Swin-MAE architecture utilizing Masked Image Modeling achieves superior feature representation. Furthermore, we introduce a Label-Free Dynamic Anomaly Suppression Loss (ASL) mechanism that triggers based on reconstruction error stagnation, preventing the model from absorbing anomaly patterns during training.

________________________________________

## 2. Introduction

### 2.1 Problem Statement
Modern agriculture relies heavily on the early detection of crop stresses to maximize yield and minimize resource wastage. Farmers face a multitude of anomalies including drydown, nutrient deficiency, weed clusters, planter skips, and water stress. However, automating this detection is computationally difficult because crops are biologically complex systems.
The core problem lies in the nature of agricultural data:
•	High Variability: Anomalies vary significantly across different crops, seasons, and field conditions.  
•	Annotation Costs: Annotating such diverse anomalies at a pixel level is extremely expensive and requires expert agronomists.  
•	Complexity: Unlike industrial datasets (e.g., MV-Tec), agricultural scenes are unstructured, contain natural randomness, and lack fixed geometry.  

Existing supervised methods struggle because anomalies are rare (leading to class imbalance) and "normal" crop appearance changes constantly. Therefore, there is an urgent need for label-free, generalizable methods that can handle the spatial and structural complexity of aerial imagery.

### 2.2 Our Contributions
Our work systematically explores and evaluates different deep learning architectures for this task. Our primary contributions are:
1.	Failure Analysis of Supervised Models: We demonstrate empirically that Conditional GANs (cGANs) struggle with the irregular boundaries of agricultural anomalies, achieving suboptimal segmentation results due to label noise and unstable adversarial training.  
2.	Identification of Reconstruction Collapse: Through the implementation of a CNN-based autoencoder, we validate that simple reconstruction objectives lead to the model "learning" the anomalies, thereby rendering them undetectable via reconstruction error maps.  
3.	Implementation of Swin-MAE: We successfully adapt the Swin Transformer with Masked Autoencoding to 4-channel satellite images (RGB + NIR), utilizing a 75% masking ratio to force the learning of global spatial contexts.  
4.	Dynamic ASL Trigger: We propose a label-free dynamic mechanism that activates Anomaly Suppression Loss (ASL) when the Mean Squared Error (MSE) of the validation set stagnates, ensuring the suppression mechanism is applied only after the model has learned the normal crop structure.  

### 2.3 System Pipeline Overview
Our final proposed system utilizes a two-stage process leveraging self-supervised learning.  
<img width="1187" height="510" alt="image" src="https://github.com/user-attachments/assets/dd9db262-5749-4673-ac1d-aafc64295fb0" />


 
Figure 1: The architecture of the Swin-MAE approach. The Encoder (green) processes visible patches using Swin Transformer blocks. The Decoder (yellow) reconstructs the full image. The difference between the input and output yields the Reconstruction Error Map.
The pipeline processes input images of size $(H, W, 4)$, handling both RGB and Near-Infrared (NIR) channels. The encoder partitions the image into patches, embeds them, and processes them through hierarchical Swin Transformer blocks. The decoder expands these patches to reconstruct the original input. Anomalies are detected by analyzing the pixel-wise difference between the input and the reconstruction.

________________________________________

## 3. Related Work
Our research builds upon foundational papers in computer vision and specific advances in anomaly detection.

### 3.1 Foundational Approaches in Vision Transformers
The shift from CNNs to Transformers in vision tasks is well-documented. Liu et al. (2021) introduced the Swin Transformer, a hierarchical vision transformer using shifted windows. This architecture is crucial for our work as it provides strong feature representation while remaining computationally efficient, making it suitable for detecting subtle deviations in high-resolution aerial images.
Furthermore, Mia et al. (2023) in "ViTs Are Everywhere" highlighted that Transformers trained with self-supervised masked modeling demonstrate robust performance in domains where anomalies lack labels, validating our choice of architecture.

### 3.2 Reconstruction-Based Anomaly Detection
Reconstruction-based detection assumes that a model trained only on normal data will fail to reconstruct anomalies. However, Arbelle et al. (2023) in MAEDAY demonstrated that while Masked Autoencoders (MAE) are powerful, deep models can become "too good" at reconstruction. Over time, the network may unintentionally learn patterns associated with anomalies, a phenomenon known as "absorption". This necessitates additional mechanisms beyond simple MSE loss.

### 3.3 Anomaly Suppression Mechanisms
To combat reconstruction collapse, Liu et al. (2023) proposed BiGSET, a separation training framework using binary masks to distinguish normal regions from suspected anomalies.
Most importantly, our work is an extension of the concepts presented by Shikhar et al. (2024) in "Label-Free Anomaly Detection in Aerial Agricultural Images with Masked Image Modeling". This paper established the Swin-Transformer-based MAE framework for agriculture and introduced a fixed Anomaly Suppression Loss (ASL). Our method extends this by making the suppression mechanism dynamic rather than fixed, aiming to improve stability during continued learning.

________________________________________

## 4. Methodology
We adopted an iterative experimental approach, testing four distinct methodologies to identify the optimal solution for agricultural anomaly detection.

### 4.1 Phase 1: Conditional GAN (Supervised)
Initially, we implemented a Conditional Generative Adversarial Network (cGAN) for supervised anomaly segmentation.  
•	Architecture: The Generator predicts pixel-wise anomaly masks from RGB+NIR images, while the Discriminator attempts to distinguish between (image, predicted mask) and (image, ground-truth mask) pairs.  
•	Hypothesis: The GAN loss would encourage sharper mask boundaries compared to standard CNNs.  
•	Outcome: The model struggled significantly. The irregular nature of agricultural anomalies, combined with class imbalance, led to unstable adversarial training. The generator often failed to converge on the complex shapes of weed clusters or water stress patches.  
•	Metric: The model achieved a Mean Intersection-Over-Union (mIoU) of only 0.1547.  

<img width="1083" height="114" alt="image" src="https://github.com/user-attachments/assets/ab363655-fdb2-4ed6-b64a-c16721caca85" />

Figure 2: Training logs for the Conditional GAN showing convergence difficulties and a low final mIoU.

### 4.2 Phase 2: CNN-Based Autoencoder (Unsupervised)
To address the lack of reliable labels, we moved to an unsupervised CNN-based autoencoder trained on the Agri-Vision Challenge Dataset.  
•	Architecture: A standard Convolutional Encoder-Decoder trained using Mean Squared Error (MSE) to reconstruct input images pixel-by-pixel.  
•	Mechanism: Ideally, the model should learn normal crop textures. When presented with an anomaly (e.g., a tractor track or dead crop), the reconstruction error should be high.  
•	Outcome - Reconstruction Collapse: The model learned both normal and abnormal patterns indiscriminately. The "reconstruction collapse" meant that anomalous regions were reconstructed almost perfectly, resulting in no error signal for detection.  
•	Metric: This approach yielded the lowest performance with an mIoU of 0.039.  

<img width="895" height="197" alt="image" src="https://github.com/user-attachments/assets/3dd4c33a-1d36-4b6c-acb3-43cbfd3c172f" />
 
Figure 3: Visualization of Reconstruction Collapse. The 'Reconstruction Error' map (third column) shows very little signal because the model successfully reconstructed the anomaly.

### 4.3 Phase 3: Swin-MAE (Self-Supervised + Fine-Tuning)
Recognizing the limitations of CNNs in global context modeling, we adopted the Swin Transformer with Masked Autoencoding.  
•	Stage 1 (Unsupervised Pretraining): We utilized a Swin Transformer Encoder to reconstruct missing patches from 4-channel satellite images. We used a masking ratio of 75%, meaning the model only saw 25% of the image and had to hallucinate the rest. This forces the model to learn high-level semantic representations of "farmland" rather than just memorizing pixels.  
•	Stage 2 (Supervised Fine-Tuning): We initialized the encoder with weights from Stage 1 and added a U-Net style segmentation head.  
•	Outcome: This method proved that self-supervised pre-training learns powerful texture representations.  
•	Metric: Achieved an mIoU of 0.2336, the highest among our supervised attempts.  

<img width="753" height="185" alt="image" src="https://github.com/user-attachments/assets/f3f0ebba-90b7-4377-89ba-6d318817bf61" />
 
Figure 4: Fine-tuning logs for Swin-MAE showing improved convergence and mIoU compared to cGAN.

### 4.4 Phase 4: Label-Free Dynamic ASL (Proposed Method)
To achieve a truly label-free system, we enhanced the Swin-MAE architecture with a Dynamic Anomaly Suppression Loss (ASL) trigger.  
•	The Logic: We track the validation reconstruction loss per epoch. Initially, the loss drops as the model learns the dominant "normal" crop features. Eventually, the learning saturates, and the loss plateaus.  
•	The Trigger: We monitor for MSE stagnation (patience = 3 epochs). Once the plateau is detected (typically around epoch 12), we activate the ASL.  
•	ASL Mechanism: The ASL down-weights pixels with high reconstruction error during the training update. This discourages the model from trying to minimize error on anomalies, effectively forcing it to "forget" or "ignore" how to reconstruct them, thereby maximizing the detection signal during testing.  
•	Outcome: This successfully separated anomalies without using ground-truth labels during the primary training phase.  
•	Metric: Achieved an unsupervised mIoU of 0.10. While numerically lower than the supervised fine-tuning, this result is significant because it requires zero pixel-level annotations.  

<img width="794" height="407" alt="image" src="https://github.com/user-attachments/assets/2d2c17e2-afc8-4eb7-b72a-82fe45e5a8cf" />

Figure 5: Validation Loss curve. The red line indicates the point of stagnation (Epoch 12) where the ASL mechanism is triggered to prevent anomaly absorption.

________________________________________

## 5. Results

### 5.1 Quantitative Results
We evaluated our models using Mean Intersection-Over-Union (mIoU), which penalizes both false positives and false negatives.  

Method	Architecture	Supervision	mIoU  
Baseline 1	CNN Autoencoder	Unsupervised	0.039  
Baseline 2	Conditional GAN	Supervised	0.155  
Method A	Swin-MAE + Fine-Tuning	Self-Supervised + Supervised	0.234  
Method B (Ours)	Swin-MAE + Dynamic ASL	Unsupervised / Label-Free	0.100  

Table 1: Comparative performance of different approaches. While fine-tuning yields higher metrics, the Dynamic ASL method offers a viable label-free alternative.

### 5.2 Qualitative Visual Analysis
The visual results highlight the strengths of the Swin-MAE approach over traditional CNNs.  

<img width="769" height="97" alt="image" src="https://github.com/user-attachments/assets/ed216949-cc42-467b-89e1-1736869e0636" />

Figure 6: Swin-MAE Results. From Left to Right: RGB Composite, Ground Truth Mask, Reconstruction Error Map, Predicted Mask. Note the high contrast in the Error Map (red), indicating successful detection of the anomaly.
In Figure 6, we observe that the Swin-MAE model generates a strong error signal (red heatmap) corresponding to the ground truth anomaly. Unlike the CNN model in Figure 3, the Swin-MAE preserves the difference between normal crops and the anomalous region.

________________________________________

## 6. Discussion

### 6.1 Why Traditional Methods Fail
Our experiments confirmed that traditional supervised ML fails on agricultural data for several reasons:  
•	Data Hunger: They require huge labeled datasets which do not exist for every specific crop/pest combination.  
•	Closed Set Limitation: Supervised models only predict classes seen during training. An unknown pest or new disease appearing in the field is often misclassified rather than flagged as an anomaly.  

### 6.2 The Problem of Reconstruction Collapse
One of the most critical insights from our work is the fatality of Reconstruction Collapse. Simple reconstruction models (like our Phase 2 CNN) quickly learn anomaly patterns. By minimizing MSE globally, the model learns to reconstruct the tractor tracks and weeds just as well as the corn. This results in a "vanishing gradient" of detection—the error map becomes flat, and the anomaly disappears.

### 6.3 Lessons from Experimentation
Through our "Journey" of four distinct phases, we derived three key lessons:  
1.	Reconstruction Alone is Not Enough: Successful models must be actively prevented from learning the anomaly distribution.  
2.	Self-Supervised Pre-training is Essential: Feature learning via MAE is essential for handling the complex textures of high-resolution aerial imagery.  
3.	Suppression-Based Learning is Necessary: Designs must shift from simple reconstruction to Suppression-Based Learning (like ASL) to ensure robust anomaly differentiation.  

________________________________________

## 7. Conclusion

### 7.1 Summary
In this report, we documented the development of an anomaly detection system for aerial agricultural images. We showed that while supervised segmentation (cGAN) is defeated by class imbalance (mIoU 0.1547), and standard autoencoders suffer from reconstruction collapse (mIoU 0.039), the integration of Swin Transformers with Masked Autoencoding provides a robust solution.
Our proposed Label-Free Dynamic ASL method effectively leverages the stagnation of training loss to trigger anomaly suppression. This allows the model to solidify its understanding of "normal" crops before being forced to reject anomalous patterns. This approach achieves an mIoU of 0.10 without any pixel-level labeling, offering a scalable solution for real-world farm monitoring.

### 7.2 Future Directions
Future work will focus on:  
1.	Refining the Trigger: Exploring more sophisticated statistical triggers for ASL activation beyond simple MSE stagnation.  
2.	Multi-Scale anomalies: Improving detection of extremely small anomalies (early-stage insects) vs. large anomalies (flooding) simultaneously.  
3.	Real-time Processing: Optimizing the Swin-MAE decoder for faster inference on edge devices (drones).  

________________________________________

## 8. References
[1] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10012-10022, 2021.  
[2] Assaf Arbelle, Leonid Karlinsky, Sivan Harary, Florian Scheidegger, Sivan Doveh, and Raja Giryes. MAEDAY: MAE for Few- and Zero-Shot Anomaly Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 5110-5119, 2023.  
[3] Md Sohag Mia, Abu Bakor Hayat Arnob, Abdu Naim, Abdullah Al Bary Voban, and Md Shariful Islam. ViTs Are Everywhere: A Comprehensive Study Showcasing Vision Transformers in Different Domains. arXiv preprint arXiv:2310.05664, 2023.  
[4] Yingjie Liu, Changhao Li, and Haibo He. BIGSET: Binary Mask-Guided Separation Training for DNN-based Hyperspectral Anomaly Detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 345-354, 2023.  
[5] Sambal Shikhar and Anupam Sobti. Label-Free Anomaly Detection in Aerial Agricultural Images with Masked Image Modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 3750-3759, 2024.
`
