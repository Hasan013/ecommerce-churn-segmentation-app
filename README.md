# 🛍️ E-Commerce Churn & Segmentation App

A powerful Streamlit app to analyze e-commerce customer behavior, **predict churn** using **machine learning** and **segment customers** for retention and revenue growth.

🔗 **Live App:** [ecommerce-churn.streamlit.app](https://ecommerce-churn.streamlit.app)

---

## 📸 App Preview  
![App Preview](./preview.png)  
A visual walkthrough of customer churn prediction, segmentation and retention strategies, all in one place.

---

## 🎯 What This App Does?

This interactive dashboard lets you:

- 🔍 Perform Exploratory Data Analysis (EDA)
- 🤖 Predict customer churn using **XGBoost** and **SHAP**
- 📊 Segment users using **RFM + KMeans clustering**
- 🎯 Identify high-risk customers
- 💡 Recommend targeted retention treatments

---

## 📁 Sample Data Requirements

Your Excel dataset should contain relevant columns like:

- **Churn** (target variable should be binary: 1 = churned, 0 = not churned)
- **OrderCount**, **CashbackAmount**, **DaySinceLastOrder**, etc.
- **Demographic/behavioral fields** like:
  - `PreferredLoginDevice`, `PreferredPaymentMode`, `Gender`, `PreferedOrderCat`
- This app **automatically handles**:
  - Missing values  
  - Feature engineering  
  - Scaling & encoding

---

## 🚀 App Sections

### 1. 🏠 Home
- Upload `.xlsx` datasets
- Smart preprocessing: missing value handling, feature engineering
- Session-based storage for seamless navigation

### 2. 📈 Exploratory Data Analysis
- Heatmaps of missing values  
- Distribution plots for numerical/categorical features  
- Churn distribution visualizations

### 3. ⚙️ Churn Prediction
- XGBoost model trained with class imbalance handling  
- Precision, Recall, F1-score at **0.5 and 0.3 thresholds**  
- SHAP-based feature importance visualization  
- **Real-time churn prediction form** for new customers

### 4. 🧩 Customer Segmentation
- RFM feature engineering (Recency, Frequency, Monetary)  
- KMeans clustering (High vs Low Value)  
- Subgroup tagging (e.g., Champions, At Risk, Loyal)  
- Churn probability overlayed on segments

### 5. 📋 Segment Summary
- Aggregated stats by segment and subgroup  
- Downloadable CSV report

### 6. 🎯 Targeted Treatment Plan
- Prescriptive treatment strategies per RFM subgroup  
- Priority-based customer targeting (High/Medium/Low)  
- Downloadable action plan  
- Visual breakdown of customer distribution by priority

---

## ⚙️ How to Use the App?

1. Upload your e-commerce dataset (.xlsx)
2. Navigate sections via sidebar:
   - EDA → Churn Prediction → Segmentation → Strategy
3. Export insights as CSV (segment summary or treatment plan)

---

## 🧑‍💻 Tech Stack

- **Frontend**: Streamlit, Plotly, Matplotlib, Seaborn
- **ML Model**: XGBoost (Churn Prediction)
- **Clustering**: KMeans, GMM (for customer segmentation)
- **Data Processing**: Pandas, Scikit-learn
- **Explainability**: SHAP (Model Insights)

---

## 👨‍💼 Built By

**Hasan Raja Khan**  
Data Analyst | Helping E-commerce Brands Reduce Churn & Boost LTV

📫 Email: [hraza9327@gmail.com](mailto:hraza9327@gmail.com)  
🔗 LinkedIn: [https://www.linkedin.com/in/hasan-raja-khan](https://www.linkedin.com/in/hasan-raja-khan)  
🌐 Portfolio: [https://hasan013.github.io/](https://hasan013.github.io/)  

---

## 🤝 Who Is This For?

Ideal for:

- 🛒 DTC Brand / Shopify Seller  
- 📦 Amazon FBA Operator  
- 📈 E-Commerce Marketing Agency  
- 💡 Product Manager in E-Com SaaS

If you are handling **churn, retention** or **LTV**, this app will help you turn customer data into growth.

---

## 🔮 Future Enhancements

- 📬 Email Automation Plugin for Churn Triggers  
- 🧪 A/B Testing Uplift Integration  
- 🧠 AutoML Hyperparameter Tuning  
- ⛲ Real-Time Data Connection (BigQuery, Snowflake, Supabase)

---

## 🧠 Need Help?

Want this customized for your store, agency or product?  

📬 **Let’s talk** → [hraza9327@gmail.com](mailto:hraza9327@gmail.com) 

🔗 **Let's connect on** → [LinkedIn](https://www.linkedin.com/in/hasan-raja-khan)
