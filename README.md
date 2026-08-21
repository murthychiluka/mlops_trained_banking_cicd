#  MLOps Models 
A machine learning project showcasing various ML models commonly used in banking applications such as fraud detection, customer churn prediction, credit scoring, customer segmentation, and mor.  
This repository demonstrates simple explanations of each model along with their typical use cases.

---

##  Machine Learning Models Used

### 1. **Logistic Regression**
**What it does:**  
Predicts binary outcomes such as yes/no, 0/1, true/false.

**Example Use Case:**  
Will the customer leave the bank? (Churn prediction)

---

### 2. **Decision Tree**
**What it does:**  
Makes decisions by asking simple if-else questions arranged in a tree structure.

**Example Use Case:**  
 If age > 30 → approv product, else → decline.

---

### 3. **Random Forest**
**What it does:**  
Builds many decision trees and combines their predictions for higher accuracy.

**Example Use Case:**  
 Fraud detection, churn prediction, credit scoring.

---

### 4. **XGBoost / LightGBM**
**What they do:**  
Boosting models that build trees sequentially while correcting previous errors. Extremely powerful for structured data.

**Example Use Case:**  
 Popular in 90% of winning Kaggle solutions.

---

### 5. **Support Vector Machine (SVM)**
**What it does:**  
Finds the best boundary that separates different classes.

**Example Use Case:**  
 Classifying emails as spam or not spam.

---

### 6. **K-Means Clustering**
**What it does:**  
Groups similar customers/items into clusters without using labels.

**Example Use Case:**  
 Customer segmentation for marketing campaigns.

---

### 7. **Naive Bayes**
**What it does:**  
Uses probability to classify text.

**Example Use Case:**  
 Spam detection, customer reviews sentiment analysis.

---

### 8. **Artificial Neural Networks (ANN)**
**What they do:**  
Brain-inspired models used for complex learning tasks.

**Example Use Case:**  
 Predicting loan defaults or customer churn.

---

### 9. **Convolutional Neural Networks (CNN)**
**What they do:**  
Excellent for image-related tasks.

**Example Use Case:**  
 Cheque image verification, ID document scanning.

---

### 10. **LSTM (Long Short-Term Memory)**
**What it does:**  
Designed for sequence and time-series data.

**Example Use Case:**  
 Predicting bank stock trends, customer transaction patterns.

---

##  Quick Summary (One-Liners)

| Model | Simple Meaning |
|-------|----------------|
| Logistic Regression | Predicts yes/no |
| Decision Tree | If-else questions |
| Random Forest | Many trees vote |
| XGBoost | Very smart trees |
| SVM | Best separating line |
| K-Means | Groups similar items |
| Naive Bayes | Probability for text |
| ANN | Brain-like model |
| CNN | Image expert |
| LSTM | Time-series expert |

---

## 📁 Project Structure (Example)


## **Project Overview**

This application takes user inputs such as age, income, loan amount, loan term, credit score, gender, and marital status, then predicts whether a loan will be **approved** or **denied**.  

It uses a **pre-trained Random Forest model** and encoders for categorical features.

---

## **Architecture**

```text
Frontend (HTML + JS)
        │
        ▼
   Flask API (/predict)
        │
        ▼
Pre-trained Random Forest Model (model_full.pkl)
        │
        ▼
Prediction returned to frontend
How it Works
1️⃣ Frontend Sends Data
User fills the form in form.html.

JavaScript collects input and sends a POST request to /predict.

Example JSON payload:

json

{
  "age": 30,
  "income": 50000,
  "loan_amount": 20000,
  "loan_term": 24,
  "credit_score": 700,
  "gender": "male",
  "married": "no"
}
2️⃣ Flask Receives the Request
python

data = request.get_json()
Converts JSON payload into a Python dictionary.

3️⃣ Flask Validates Input
Ensures all required fields are present: age, income, loan_amount, loan_term, credit_score, gender, married.

Returns an error if fields are missing or extra.

4️⃣ Encode Categorical Variables
Converts "male"/"female" and "yes"/"no" to numerical values using LabelEncoder.

5️⃣ Prepare Features for the Model
python

input_features = np.array([[age, income, loan_amount, loan_term, credit_score, gender_encoded, married_encoded]])
Converts input into the 2D array format expected by the Random Forest model.

6️⃣ Make Prediction
python

prediction = model.predict(input_features)
result = le_approved.inverse_transform(prediction)[0]
The model predicts 0 or 1.

Encoders convert it back to "approved" or "denied".

7️⃣ Send Response Back to Frontend
python

return jsonify({"prediction": result, "status": "success"})
Frontend receives JSON and displays the result:

text

✅ Loan Status: APPROVED






##### Backedn and model communication #####


1️⃣ model.predict(input_features)

model is your pre-trained Random Forest model (loaded from model_full.pkl at the start of app.py).

input_features is a 2D NumPy array containing one sample from the frontend (age, income, loan_amount, loan_term, credit_score, gender_encoded, married_encoded).
Example:

input_features = np.array([[30, 50000, 20000, 24, 700, 1, 0]])


Calling model.predict(input_features) runs the Random Forest model in memory and returns a numeric prediction.
For your case:

0 → denied

1 → approved

2️⃣ le_approved.inverse_transform(prediction)

Your model predicts numbers (0 or 1) but the frontend expects readable labels: "approved" or "denied".

le_approved.inverse_transform() converts the numeric output back into the string label.

Example:

prediction = np.array([1])
result = le_approved.inverse_transform(prediction)[0]  # "approved"

3️⃣ return jsonify(...)

Flask converts the result into a JSON response that goes back to the frontend.

Example JSON returned:

{
  "prediction": "approved",
  "status": "success"
}

✅ Summary

Frontend sends user input → JSON.

Flask reads the JSON → prepares input_features.

Flask passes input_features to trained Random Forest model → model.predict().

Encoders convert numeric output to "approved"/"denied".

Flask returns JSON → frontend displays it.
