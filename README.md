# Deepfake Detection with ConvNeXt V2

This project fine-tunes ConvNeXt V2 to detect deepfake visual content, achieving up to **98.8% F1-score** after applying t-SNE for error analysis and threshold tuning.

## Background

With the rapid advancement of deepfake generation technologies, malicious actors have increasingly leveraged them for scams and misinformation. This project aims to develop a reliable deepfake detection pipeline using computer vision to enhance digital security.

## Highlights
- Model: ConvNeXt V2 (image classification)
- Performance: 93.8% → **98.8% F1-score** after iterative error analysis
- Techniques: t-SNE embedding visualization, ROC/PR curve analysis, threshold tuning
- Practical focus: Evaluation and visualization workflows designed to support real-world model decisions.
- Business context: Built as part of an internal AI initiative for real-world application

## Evaluation and Error Analysis

To refine model performance, we used multiple evaluation techniques:

- **t-SNE feature embedding visualization** – to identify ambiguous samples and misclassification patterns
- **Threshold tuning** – optimized decision boundaries based on distribution of prediction scores
- **ROC and PR curve analysis** – used to compare baseline and tuned models in terms of precision and recall trade-offs

Detailed implementation is integrated into the training and evaluation workflow.

## Repository Overview

| File                     | Description |
|--------------------------|-------------|
| `eval_by_threshold.ipynb` | Threshold-based evaluation of model performance |
| `tsne.ipynb`             | Visualizes feature embeddings and error clustering |
| `draw_roc_curve.py`      | Plots ROC curves for model comparison |
| `predict.py`             | Inference script for new images |
| `requirements.txt`       | Required Python packages |

## Future Work
To improve detection accuracy and robustness, additional research is ongoing to cover various deepfake generation techniques and apply model ensemble strategies.

## Sample Result
### t-SNE Visualization: Before and After Threshold Tuning

We applied t-SNE to visualize the feature embeddings of real and fake samples before and after model tuning. This visualization helped identify overlapping clusters and guided threshold adjustments that significantly improved performance.

- **Before Tuning**  
  Clusters of real and fake images were overlapping, making classification ambiguous.

  ![image](https://github.com/user-attachments/assets/9f7341b8-dd48-4cc5-ba14-0db8f97321a0)

- **After Tuning**  
  The updated threshold separated real and fake embeddings more distinctly, contributing to a final F1-score of **98.8%**.

  ![image](https://github.com/user-attachments/assets/6b1e41c1-33fe-4d2c-9cb1-1028494969b3)
