# Deepfake Detection with ConvNeXt V2

This project implements a deepfake detection pipeline using ConvNeXt V2, achieving up to **98.8% F1-score** through rigorous feature analysis and threshold tuning.  
**This repository is based on a production model pipeline. Some components have been removed or refactored for privacy compliance, but the core methodology and structure remain faithful to the original.**

---

## Background

With the rapid growth of deepfake generation tools, malicious actors increasingly exploit them in scams and misinformation. This project aims to build a robust deepfake visual detection system that can distinguish between real and fake image content.

---

## Highlights

- **Model**: ConvNeXt V2 (fine-tuned on internal dataset)
- **Performance**: Improved F1-score from 93.8% to **98.8%**
- **Techniques**:  
  - t-SNE feature embedding visualization  
  - Threshold tuning based on score distributions  
  - ROC / PR curve analysis
- **Tools**: PyTorch, Transformers, ONNX, OpenCV, W&B (for training tracking)
- **Application**: Built for integration into real-world cybersecurity product pipelines

---

## Pipeline Overview

1. **Data Preprocessing**
   - Extract video frames using multithreading (`extract_frames_from_video.py`)
   - Detect and crop faces from frames using ONNX-based RetinaFace model (`extract_faces_from_frames.py`)

2. **Model Training**
   - Fine-tune ConvNeXt V2 using HuggingFace Trainer API
   - Track performance and perform hyperparameter tuning via W&B

3. **Evaluation**
   - Visualize learned embeddings using t-SNE
   - Tune thresholds and analyze ROC/PR curves to refine classification boundaries

---

## Repository Overview

| File / Folder                            | Description                                                      |
|------------------------------------------|------------------------------------------------------------------|
| `data_preprocessing/`                    | Scripts for frame extraction and face detection                  |
| ├── `extract_frames_from_video.py`       | Extracts 10 frames from each `.mp4` video                        |
| ├── `extract_faces_from_frames.py`       | Uses ONNX model to detect and crop faces from frames             |
| `model_training_and_tuning.ipynb`        | ConvNeXt V2 training and W&B-based hyperparameter tuning         |
| `tsne.ipynb`                             | t-SNE embedding visualization and error analysis                 |
| `requirements.txt`                       | Required Python packages                                         |

---

## Sample Results

### t-SNE Visualization (Before vs After Threshold Tuning)

We applied t-SNE to visualize the feature embeddings of real and fake samples before and after model tuning. This visualization helped identify overlapping clusters and guided threshold adjustments that significantly improved performance.

- **Before Tuning**  
  Clusters of real and fake images were overlapping, making classification ambiguous.

  ![image](https://github.com/user-attachments/assets/9f7341b8-dd48-4cc5-ba14-0db8f97321a0)

- **After Tuning**  
  The updated threshold separated real and fake embeddings more distinctly, contributing to a final F1-score of **98.8%**.

  ![image](https://github.com/user-attachments/assets/6b1e41c1-33fe-4d2c-9cb1-1028494969b3)

---

## Note on Privacy

Some deployment-related scripts and intermediate outputs have been omitted from this repository due to internal security policies. However, all code structures and methodology remain representative of the original system.

---

## Future Work

- To improve detection accuracy and robustness, additional research is ongoing to cover various deepfake generation techniques and apply model ensemble strategies.
