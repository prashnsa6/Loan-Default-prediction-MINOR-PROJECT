# 🧠 Loan Default Prediction – Logistic Regression Mini Project
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


This repository presents an end-to-end machine learning workflow for predicting **loan default risk** using **Logistic Regression**.  
The project is based on a **public Kaggle dataset** and demonstrates the complete implementation process — from **data exploration** and **preprocessing** to **model training**, **evaluation**, and **interpretation**.

---

## 🎯 Objective

The goal of this mini-project is to build a predictive model that determines whether a borrower is likely to **default on a loan**, using demographic, financial, and loan-related variables such as:

- Credit type  
- Loan-to-Value ratio (LTV)  
- Loan purpose  
- Upfront charges  
- Property and income details  

---

## ⚙️ Project Workflow

1. **Exploratory Data Analysis (EDA)**  
   Examined missing values, distributions, and class imbalance.  
2. **Data Preprocessing**  
   - Imputation (median for numeric, constant for categorical)  
   - Feature scaling with `StandardScaler`  
   - One-Hot Encoding for categorical variables  
3. **Model Training**  
   - Logistic Regression (`saga` solver, `class_weight='balanced'`)  
4. **Model Evaluation**  
   - ROC-AUC, Average Precision, Recall, Precision, and F1-score  
   - Confusion Matrix, ROC and PR curves  
5. **Feature Interpretation**  
   - Extracted and ranked coefficients to identify top predictors.  
6. **Model Saving**  
   - Trained pipeline and metrics saved to `/outputs/`

---

## 📈 Model Performance

| Metric | Score |
|:--|:--:|
| **ROC-AUC** | 0.867 |
| **Average Precision (AP)** | 0.799 |
| **Recall (Class 1)** | 0.710 |
| **Precision (Class 1)** | 0.661 |
| **F1-Score (Class 1)** | 0.685 |

✅ The model performs well, showing strong discriminatory capability and balanced recall and precision for the default class.

---

## 🔍 Top Predictive Features

| Rank | Feature | Coefficient | Interpretation |
|:--:|:--|:--:|:--|
| 1 | `credit_type_EQUI` | +6.45 | Greatly increases probability of default |
| 2 | `credit_type_EXP` | −1.97 | Decreases default probability |
| 3 | `credit_type_CIB` | −1.93 | Decreases default probability |
| 4 | `lump_sum_payment_lpsm` | +1.60 | Increases default risk |
| 5 | `LTV` | +0.87 | Higher loan-to-value → higher risk |
| 6 | `Upfront_charges` | −0.55 | Higher upfront payments reduce risk |

📌 **Positive coefficient** → increases chance of default  
📌 **Negative coefficient** → decreases chance of default  

---

## 🗂️ Repository Structure

Loan-Default-Prediction/
├── data/ # (optional) dataset folder
├── notebooks/
│ └── Loan_Default_Prediction.ipynb
├── outputs/
│ ├── metrics.json # evaluation metrics
│ ├── top_coeffs.csv # feature coefficients
│ ├── roc_curve.png # ROC curve
│ ├── pr_curve.png # Precision-Recall curve
│ └── logreg_pipeline.joblib # saved model
├── scripts/
│ ├── train_logreg_full.py # main training script (command-line)
│ └── train_logreg_full.ipynb # Jupyter Notebook version of training workflow
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md


---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/Loan-Default-Prediction.git
cd Loan-Default-Prediction

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model
python scripts/train_logreg_full.py --zip "data/Loan_Default.csv.zip" \
    --csvname "Loan_Default.csv" \
    --target "Status" \
    --top_k 50 \
    --max_iter 3000

4️⃣ View results

Model outputs appear in the /outputs/ directory:

metrics.json → Evaluation metrics

roc_curve.png, pr_curve.png → Model performance plots

top_coeffs.csv → Feature coefficients

logreg_pipeline.joblib → Trained model pipeline

📊 Example Results

Confusion Matrix

Actual / Predicted	No Default	Default
No Default		      19740		      2666
Default			      2122		      5206

🧭 Next Steps

Tune Logistic Regression regularization (C) with GridSearchCV.

Apply k-fold cross-validation for robustness.

Explore ensemble models such as Random Forest or XGBoost.

Use SHAP or LIME for model explainability.

---

## 👩‍💻 Author

**Prasansa Vanjari**  
Developed as part of a certification course project on machine learning and data analytics.  
This repository demonstrates an end-to-end workflow for loan default prediction using Logistic Regression.

---

## 📄 License

This project is licensed under the terms of the **MIT License**.  
See the [LICENSE](LICENSE) file for the complete license text.

> **Educational Disclaimer:**  
> This project was developed as part of an academic certification course and is intended **solely for educational and learning purposes**.  
> The dataset used is **publicly available**, and this work is **not intended for commercial use**.