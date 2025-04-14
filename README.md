# 🧠 Bank Customer Churn Prediction (Python - KNN, SVM & Random Forest)

This project uses machine learning to **predict whether bank customers are likely to churn** (leave the bank) based on key features such as balance, tenure, credit card status, and more.

The goal is to **support proactive retention strategies** by identifying at-risk customers using **K-Nearest Neighbors (KNN)**, **Support Vector Machine (SVM)**, and **Random Forest** models.

---

## 📌 Project Highlights

- 🔍 Cleaned and preprocessed real-world churn data using Python
- 📊 Performed exploratory data analysis to uncover churn trends
- 🤖 Trained and evaluated KNN, SVM, and Random Forest models
- 📈 Exported structured prediction results for use in BI dashboards

---

## 📁 Project Structure

```
bank-churn-prediction-ml/
│
├── BankChurn.ipynb           # Main Jupyter notebook with all logic
├── churn_predictions.csv     # Exported predictions (for BI use)
├── requirements.txt          # Python dependencies
└── README.md                 # You're here!
```

---

## 🧹 Data Preprocessing

- Handled missing values
- Encoded categorical features
- Standardized numerical values
- Split into train/test sets

Clean data = better predictions & easier visualization later in Power BI.

---

## 📊 Exploratory Data Analysis (EDA)

Key insights:
- Churn is higher among **short-tenure** customers
- **Germany** has a notably higher churn rate
- **Females** churn more than males in this dataset
- Customers **without credit cards** show higher churn

Visualizations include bar plots, heatmaps, and box plots using `matplotlib` and `seaborn`.

---

## 🤖 Machine Learning Models

Three models were built and evaluated to predict customer churn:

### 🔹 K-Nearest Neighbors (KNN)
- Simple, instance-based model
- Tuned the `k` value based on accuracy

### 🔹 Support Vector Machine (SVM)
- Effective in high-dimensional spaces
- Tuned using kernel options (e.g., linear, RBF)

### 🔹 Random Forest Classifier
- Ensemble model that builds multiple decision trees
- Strong performance on structured/tabular data
- Provides feature importance for better interpretability

  ### 🛠️ Hyperparameter Tuning
Hyperparameter tuning was applied to improve model performance for each algorithm:

- **KNN**: Optimal `k` value was selected based on accuracy scoring
- **SVM**: Kernel type and regularization parameter `C` were tuned using grid search
- **Random Forest**: Number of trees (`n_estimators`), maximum depth, and minimum samples per split were optimized

All tuning was performed using `GridSearchCV` to find the best parameter combinations on cross-validation.

### 📈 Metrics Reported:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

All results were printed in a structured performance report and compared side-by-side to evaluate model effectiveness.

---

## 📤 Exporting Predictions

The final predictions (including probabilities) were saved as a `.csv` for external use in **Power BI dashboards** or other analytics tools.

Sample columns:
```
CustomerID | PredictedChurn | ChurnProbability | Gender | Geography | Balance
```

---

## 🗃️ Data Source

This dataset was provided by **[Maven Analytics](https://www.mavenanalytics.io/data-playground)** as part of their free public data projects.

- Dataset Name: *Bank Customer Churn*
- Data includes 10,000 bank customers from a European financial institution

---

## 🛠 Tech Stack

- 🐍 Python 3
- 📦 pandas, numpy
- 📊 matplotlib, seaborn
- 🤖 scikit-learn

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bank-churn-prediction-ml.git
   cd bank-churn-prediction-ml
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook BankChurn.ipynb
   ```

4. Run all cells to preprocess data, train models, and view performance.

---

## 📌 Next Steps

- Perform hyperparameter tuning
- Deploy model as an API using Flask or FastAPI
- Integrate with Power BI service for live predictions

---

## 🙋 About Me

I'm passionate about data science, analytics, and solving real-world problems using machine learning. Let's connect!

- 🔗 [LinkedIn]([#](https://www.linkedin.com/in/halah-almekhlafi--303395134/))
- 🐙 [GitHub](#)

---

## ⭐️ If you like this project...

Please consider giving it a ⭐️ and following me on GitHub!
