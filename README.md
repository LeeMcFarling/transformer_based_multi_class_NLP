# **HuffPost News Category Classification with Transformers**

A multi-class natural language processing (NLP) pipeline for automatically classifying HuffPost news articles into 22 editorial categories using a fine-tuned DistilBERT transformer.

This project was completed as part of Boston University's **DX703: Advanced Machine Learning & AI** course. The objective was to develop an end-to-end deep learning pipeline capable of accurately categorizing short-form news articles while addressing real-world challenges including class imbalance, ambiguous labels, and overlapping semantic categories.

---

## Project Overview

Modern news organizations publish thousands of articles each day, making manual topic assignment both time-consuming and inconsistent. This project develops a transformer-based classification system capable of automatically assigning editorial categories using only article headlines and short descriptions.

Key challenges addressed include:

- Multi-class classification across 22 news categories
- Highly imbalanced class distributions
- Short and ambiguous text inputs
- Semantically overlapping categories
- Efficient fine-tuning of pretrained language models

The final model achieved approximately **71.8% test accuracy** with a **macro F1-score of 0.671**, demonstrating strong performance while maintaining balanced prediction quality across minority classes.

---

## Repository Structure

```
.
├── README.md
├── Huffpost Classification.pdf        # Technical report
├── experimental_Mod_7_Final_Project.ipynb
└── images/
    ├── category_distribution.png
    ├── impact_workflow.png
    ├── training_curves.png
    └── confusion_matrix.png
```

---

## Methodology

The complete machine learning pipeline includes:

### Data Preparation

- Combined headlines and short descriptions into a unified text feature
- Removed duplicates and empty observations
- Standardized formatting and whitespace
- Consolidated 41 original HuffPost categories into 22 broader editorial classes
- Applied stratified 80/10/10 train-validation-test splits

### Feature Engineering

- DistilBERT tokenizer
- Maximum sequence length of 128 tokens
- Integer label encoding
- Balanced class weighting for imbalanced categories

### Model Architecture

The final classifier consists of:

- Pretrained DistilBERT encoder
- Custom fully-connected classification head
    - Dense (1028)
    - Dense (512)
    - Dense (256)
- Dropout regularization
- Softmax output across 22 categories

### Training Strategy

A two-stage fine-tuning process was used:

1. Freeze the transformer backbone and train only the custom classifier.
2. Unfreeze the full model for end-to-end fine-tuning using a reduced learning rate.

This approach provided more stable optimization while reducing overfitting.

---

## Results

Final model performance:

| Metric | Score |
|---------|------:|
| Test Accuracy | **71.8%** |
| Macro F1 | **0.671** |
| Classes | 22 |
| Dataset Size | ~200,000 articles |

Additional evaluation included:

- Confusion matrix analysis
- Per-class precision, recall, and F1 scores
- Training and validation learning curves
- Comparison against an initial baseline model

A notable contribution of the project was reassigning the original **Impact** category into semantically appropriate classes before retraining, improving minority-class performance and overall macro F1.

---

## Technologies

- Python
- TensorFlow / Keras
- Hugging Face Transformers
- DistilBERT
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

---

## Key Takeaways

This project illustrates several practical techniques used in modern NLP workflows:

- Transfer learning with pretrained transformers
- Two-stage fine-tuning strategies
- Handling severe class imbalance
- Label engineering to improve downstream performance
- Comprehensive evaluation beyond overall accuracy

The resulting classifier demonstrates how transformer models can effectively organize large collections of short-form news content with relatively modest computational resources.

---

## Documentation

The repository includes:

- **Technical Report (PDF)** — detailed methodology, evaluation, and discussion.
- **Jupyter Notebook** — complete preprocessing, model development, experimentation, and visualization pipeline.

---

## Future Improvements

Potential extensions include:

- RoBERTa or DeBERTa architectures
- Focal loss for minority classes
- Data augmentation for short text
- Domain-specific pretraining on news corpora
- Hyperparameter optimization
- Ensemble transformer models

---

## Author

**Lee McFarling**

M.S. Data Science — Boston University

This project was completed as part of the graduate coursework for **DX703: Advanced Machine Learning & AI**.