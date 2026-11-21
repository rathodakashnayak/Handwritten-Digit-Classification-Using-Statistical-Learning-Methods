# Handwritten Digit Classification using Statistical Learning Methods

**Statistical Learning I – Course Project** · Kaggle [Digit Recognizer](https://www.kaggle.com/competitions/digit-recognizer) (MNIST)

> Can the 784 pixel intensities of a 28×28 image identify the digit it shows, and which statistical learning method does it best?

---

## 👥 Team

| Name | Roll Number | Responsibility |
|------|-------------|----------------|
| Rathod Akash | BSD-BG-2415 | End-to-end execution |
| Adarsh | BSD-DH-2403 | Team member |
| Maurya | BSD-DH-2417 | Team member |
| Tabish | BSD-DH-2423 | Team member |
| Sarvesh | BSD-DH-2416 | Team member |

---

## 📌 Overview

This project classifies handwritten digits (0–9) from raw pixel values and compares classical statistical learning methods using a leakage-aware workflow:

data checks → EDA → stratified 80:20 split → baseline → 7 classifiers + PCA → cross-validated tuning → diagnostics → bias–variance analysis → final model → Kaggle predictions.

- **Data:** 42,000 labelled training images (784 pixel predictors, 10 classes) and 28,000 unlabelled test images
- **Validation:** 33,600 training / 8,400 held-out validation images (stratified, `random_state=42`)
- **Primary metric:** accuracy (also macro precision, recall, F1 and confusion matrix)

![Workflow](images/e_workflow.png)

---

## 🔑 Key Results

| Model | Validation accuracy | Macro F1 |
|-------|:------------------:|:--------:|
| **PCA (60) + RBF SVM** | **0.9710** | 0.9706 |
| k-NN (k = 5, distance) | 0.9677 | 0.9675 |
| Random forest (250 trees) | 0.9660 | 0.9655 |
| Logistic regression (L2) | 0.9130 | 0.9118 |
| L1 logistic regression | 0.9112 | 0.9100 |
| LDA (shrinkage) | 0.8731 | 0.8720 |
| Decision tree | 0.8551 | 0.8534 |
| Majority-class baseline | 0.1115 | – |

**Tuned models (3-fold CV on a 10,000-image subset), evaluated on the validation set:**

| Model | Validation accuracy | CV accuracy |
|-------|:------------------:|:-----------:|
| **Tuned random forest** (selected by the notebook) | **0.9525** | 0.9469 |
| Tuned k-NN | 0.9130 | 0.9081 |
| Tuned logistic regression | 0.9082 | 0.9084 |
| Tuned decision tree | 0.8118 | 0.7957 |

The selected tuned random forest (350 trees, `max_depth=25`, `max_features='sqrt'`) was refitted on all 42,000 labelled images to predict the 28,000 Kaggle test images.

![Model comparison](images/acc_bar.png)

---

## 📊 Exploratory Data Analysis

- No missing values, no duplicate images; all ten digits present (9.0 % – 11.2 % each) → classes are nearly balanced.
- 76 of 784 pixels have zero variance; information is concentrated in a central oval region.
- Pixel intensities are bimodal: mostly background (0) with ink near 255.
- 153 principal components retain 95 % of the variance (87 retain 90 %).

![Class distribution](images/class_dist.png)
![Mean images](images/mean_images.png)
![Pixel variance](images/pixel_variance.png)
![PCA explained variance](images/pca_cumvar.png)

---

## 🔍 Diagnostics

- The best initial model (PCA + SVM) makes **244 errors out of 8,400 (2.90 %)**.
- Most frequent confusions: **4↔9, 3↔5, 3↔2, 8↔9** – visually similar shapes; many errors are genuinely ambiguous handwriting.
- Digit **1** is easiest (F1 ≈ 0.989); digit **9** is hardest (F1 ≈ 0.955).
- A single decision tree over-fits strongly (train 100 %, validation 85.5 %, gap 14.5 points); the SVM generalises best (gap 0.7 points).

![Confusion matrix – SVM](images/cm_svm.png)
![Misclassified examples](images/misclassified.png)
![Train vs validation](images/e_gap.png)
![Learning curve](images/learning_curve.png)
![Random forest pixel importance](images/rf_importance.png)

---

## ⚠️ Important Observation

The selected tuned random forest (**95.25 %**) is lower than three untuned models on the same validation set (SVM 97.10 %, k-NN 96.77 %, forest 96.60 %). Reasons visible in the notebook:

1. Tuning used only a **10,000-image subset** (~30 % of the training split).
2. The **SVM was never tuned** or entered in the selection stage.
3. The logistic and k-NN pipelines also added `StandardScaler` (a likely, untested contributor to the k-NN drop).

**Recommendation:** tune the PCA + SVM pipeline (C, gamma, PCA dimension) on the full training split and select among all candidates. Differences between the top three models are also small: the standard error of accuracy on 8,400 images is about 0.19 percentage points.

![Before vs after tuning](images/e_before_after.png)

---

## 🗂️ Repository Structure

```
.
├── Statistical_Learning_Digit_Recognizer_Project_.ipynb   # Full notebook (Google Colab)
├── Digit_Recognizer_Report.pdf                            # Project report
├── Digit_Recognizer_Report.tex                            # LaTeX source (needs figs/)
├── images/                                                # Figures used in this README
└── README.md
```

## ▶️ How to Run

1. Open the notebook in **Google Colab**.
2. Download `train.csv` and `test.csv` from the [Kaggle competition page](https://www.kaggle.com/competitions/digit-recognizer) (or upload `kaggle.json` to use the Kaggle API).
3. Upload the files when prompted and run all cells from top to bottom.
4. Runtime depends on Colab hardware; L1 logistic regression takes about 5 minutes.

**Libraries:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

> `test.csv` has no labels, so test accuracy cannot be computed locally. The honest performance estimate is the held-out validation accuracy.

---

## 🚀 Future Work

- Tune the PCA + SVM pipeline on the full training data and re-select
- Compare PCA dimensions (30, 60, 100, 150)
- Regularised QDA; linear vs RBF SVM
- Soft-voting ensemble (SVM + k-NN + forest)
- Small MLP / convolutional network
- Probability calibration
- Run the personal handwritten-digit demo (code present in the notebook but commented out)

---

## 📚 References

- LeCun, Bottou, Bengio & Haffner (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*.
- Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*. Springer.
- James, Witten, Hastie & Tibshirani (2021). *An Introduction to Statistical Learning*. Springer.
- Cortes & Vapnik (1995). Support-vector networks. *Machine Learning*.
- Breiman (2001). Random forests. *Machine Learning*.
- Pedregosa et al. (2011). Scikit-learn: Machine learning in Python. *JMLR*.
