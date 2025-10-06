# credit-_card-_transaction_fraud_detection
This project detects fraudulent transactions using Logistic Regression and Random Forest. Data preprocessing included encoding, scaling, and SMOTE for class imbalance. Random Forest with SMOTE achieved the best performance based on accuracy, precision, recall, and F1-score.
# 🧠 Fraud Detection Using Machine Learning

## 📌 Overview
This project aims to detect **fraudulent financial transactions** using machine learning models. Fraud detection is a critical challenge due to the **imbalance between fraud and non-fraud cases**. To address this, we applied **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset and used **Logistic Regression** and **Random Forest** for classification.

---

## 📊 Dataset
The dataset (`fraud_data.xlsx`) includes:
- **TransactionID:** Unique transaction identifier  
- **Amount:** Transaction amount  
- **Time:** Time of transaction  
- **Location:** Transaction location (categorical)  
- **MerchantCategory:** Type of merchant (categorical)  
- **CardHolderAge:** Age of cardholder  
- **IsFraud:** Target variable (1 = Fraud, 0 = Genuine)

---

## ⚙️ Data Processing
1. **Encoding:** Converted `Location` and `MerchantCategory` into numeric form using one-hot encoding.  
2. **Scaling:** Standardized features using `StandardScaler`.  
3. **Balancing:** Applied **SMOTE** to handle class imbalance.  
4. **Exploration:** Used heatmaps to study correlations between variables.

---

## 🤖 Model Building
Two models were trained and evaluated:

- **Logistic Regression:** Served as the baseline model; achieved high accuracy but failed to identify many fraud cases.  
- **Random Forest:** Improved detection using SMOTE and `class_weight='balanced'`, resulting in better recall and balanced performance.

```python
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_model.fit(X_train_resampled, y_train_resampled)
