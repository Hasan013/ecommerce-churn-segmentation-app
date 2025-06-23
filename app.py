import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# App configuration
st.set_page_config(page_title="E-Commerce Churn & Segmentation", layout="wide")
st.title("E-Commerce Churn & Segmentation App")

# Sidebar navigation
section = st.sidebar.selectbox("Select Section", ["Home", "Exploratory Data Analysis", "Churn Prediction", "Customer Segmentation", "Segment Summary", "Targeted Treatment"])

# Preprocessing function
@st.cache_data
def preprocess_data(df):
    try:
        df_clean = df.drop(columns=['CustomerID'], errors='ignore')
        df_clean['PreferredLoginDevice'] = df_clean['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})
        df_clean['PreferredPaymentMode'] = df_clean['PreferredPaymentMode'].replace({'CC': 'Credit Card', 'COD': 'Cash on Delivery'})
        df_clean['PreferedOrderCat'] = df_clean['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})
        df_clean['AvgOrderValue'] = df_clean['CashbackAmount'] / df_clean['OrderCount'].replace(0, 1)
        df_clean['RecentComplaint'] = (df_clean['Complain'] == 1).astype(int)
        df_clean = df_clean.drop_duplicates()
        missing_cols = df_clean.columns[df_clean.isnull().mean() > 0]
        if missing_cols.size > 0:
            iter_cols = [col for col in missing_cols if col != 'HourSpendOnApp' and col in df_clean.select_dtypes(include=['int64', 'float64']).columns]
            if iter_cols:
                iter_imputer = IterativeImputer(random_state=42)
                df_clean[iter_cols] = iter_imputer.fit_transform(df_clean[iter_cols])
            if 'HourSpendOnApp' in missing_cols:
                simple_imputer = SimpleImputer(strategy='median')
                df_clean['HourSpendOnApp'] = simple_imputer.fit_transform(df_clean[['HourSpendOnApp']])
        return df_clean
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None

# Home Section
if section == "Home":
    st.header("Home")
    st.markdown("Welcome! This helps you analyze customer churn and segment your e-commerce audience using machine learning and RFM analysis.")
    st.markdown("Upload your E-Commerce dataset (Excel) to explore churn prediction and customer segmentation.")
    uploaded_file = st.file_uploader("Upload E-Commerce Dataset (Excel)", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=1)
            st.session_state['df'] = df
            st.write("Dataset Preview:")
            st.dataframe(df.head(5))
            df_clean = preprocess_data(df)
            if df_clean is not None:
                st.session_state['df_clean'] = df_clean
                st.success("Data preprocessed successfully! Navigate to other sections.")
            else:
                st.error("Preprocessing failed. Check dataset format.")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    else:
        st.warning("Please upload the dataset to proceed.")
    st.markdown("""
    <div style="text-align: left; font-size: 18px;">
        👤 Created by Hasan Raja Khan - Data Analyst
    </div>
    """, unsafe_allow_html=True)

# Exploratory Data Analysis Section
if section == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    if 'df' in st.session_state:
        df = st.session_state['df']
        # Missing Values Heatmap
        st.subheader("Missing Values Heatmap")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        st.pyplot(fig)

        # Numerical Features Distribution
        st.subheader("Numerical Features Distribution")
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        col = st.selectbox("Select Numerical Column", numerical_cols)
        fig = px.box(df, y=col, title=f"Boxplot of {col}")
        fig.update_layout(width=600, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Correlation Matrix
        st.subheader("Correlation Matrix")
        if numerical_cols.size > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(df[numerical_cols].corr(), annot=False, cmap='coolwarm')
            plt.title('Correlation Matrix')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("No numerical columns available for correlation matrix.")

        # Categorical Features vs Churn
        st.subheader("Categorical Features vs Churn")
        categorical_cols = df.select_dtypes(include=['object']).columns
        if categorical_cols.size > 0:
            selected_cat = st.selectbox("Select Categorical Column", categorical_cols)
            fig = px.histogram(df, x=selected_cat, color='Churn', barmode='group', title=f'{selected_cat} vs Churn')
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns available.")

        # Churn Distribution
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        fig = px.pie(values=churn_counts.values, names=churn_counts.index, title='Churn Distribution')
        fig.update_layout(width=400, height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please upload the dataset on the Home page.")

# Churn Prediction Section
if section == "Churn Prediction":
    st.header("Churn Prediction")
    if 'df_clean' in st.session_state:
        df_clean = st.session_state['df_clean']
        try:
            X = df_clean.drop(columns=['Churn'])
            y = df_clean['Churn']
            # Churn Class Distribution
            st.subheader("Churn Class Distribution")
            churn_counts = y.value_counts()
            st.write(churn_counts)
            scale_pos_weight = churn_counts[0] / churn_counts[1] if churn_counts[1] > 0 else 1
            # Dummy encoding
            X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=True)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            # Scale numerical features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            @st.cache_resource
            def train_xgboost():
                model = XGBClassifier(
                    learning_rate=0.1,
                    max_depth=5,
                    n_estimators=200,
                    random_state=42,
                    eval_metric='logloss',
                    scale_pos_weight=scale_pos_weight
                )
                model.fit(X_train, y_train)
                return model
            model = train_xgboost()
            st.session_state['best_model'] = model
            st.session_state['X_columns'] = X.columns
            st.session_state['scaler'] = scaler

            # Model Evaluation
            st.subheader("Model Evaluation (XGBoost Classifier)")
            st.markdown("""
            *Note: The model predicts 'Churn' based on a probability threshold. A threshold of 0.5 is the standard default, balancing precision and recall. A threshold of 0.3 is used to increase recall, identifying more at-risk customers in this imbalanced dataset (~15% churn).*
            """)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            # Summarize key metrics in a single line
            report = classification_report(y_test, y_pred, output_dict=True)
            st.write(f"**Threshold=0.5**: Precision (Churn): {report['1']['precision']:.2f}, Recall (Churn): {report['1']['recall']:.2f}, F1-Score (Churn): {report['1']['f1-score']:.2f}, Accuracy: {report['accuracy']:.2f}")
            
            # Tabular Classification Report for Threshold=0.5
            st.write("**Classification Report (Threshold=0.5)**:")
            report_df = pd.DataFrame({
                'Class': ['No Churn (0)', 'Churn (1)'],
                'precision': [report['0']['precision'], report['1']['precision']],
                'recall': [report['0']['recall'], report['1']['recall']],
                'f1-score': [report['0']['f1-score'], report['1']['f1-score']],
                'support': [report['0']['support'], report['1']['support']],
                'accuracy': [report['accuracy'], report['accuracy']],
                'macro avg': [report['macro avg']['f1-score'], report['macro avg']['f1-score']],
                'weighted avg': [report['weighted avg']['f1-score'], report['weighted avg']['f1-score']]
            })
            report_df[['precision', 'recall', 'f1-score', 'accuracy', 'macro avg', 'weighted avg']] = report_df[['precision', 'recall', 'f1-score', 'accuracy', 'macro avg', 'weighted avg']].round(2)
            report_df['support'] = report_df['support'].astype(int)
            st.dataframe(report_df, use_container_width=True)

            custom_threshold = 0.3
            y_pred_custom = (y_pred_proba >= custom_threshold).astype(int)
            report_custom = classification_report(y_test, y_pred_custom, output_dict=True)
            st.write(f"**Threshold=0.3**: Precision (Churn): {report_custom['1']['precision']:.2f}, Recall (Churn): {report_custom['1']['recall']:.2f}, F1-Score (Churn): {report_custom['1']['f1-score']:.2f}, Accuracy: {report_custom['accuracy']:.2f}")
            
            # Tabular Classification Report for Threshold=0.3
            st.write(f"**Classification Report (Threshold={custom_threshold})**:")
            report_custom_df = pd.DataFrame({
                'Class': ['No Churn (0)', 'Churn (1)'],
                'precision': [report_custom['0']['precision'], report_custom['1']['precision']],
                'recall': [report_custom['0']['recall'], report_custom['1']['recall']],
                'f1-score': [report_custom['0']['f1-score'], report_custom['1']['f1-score']],
                'support': [report_custom['0']['support'], report_custom['1']['support']],
                'accuracy': [report_custom['accuracy'], report['accuracy']],
                'macro avg': [report_custom['macro avg']['f1-score'], report_custom['macro avg']['f1-score']],
                'weighted avg': [report_custom['weighted avg']['f1-score'], report_custom['weighted avg']['f1-score']]
            })
            report_custom_df[['precision', 'recall', 'f1-score', 'accuracy', 'macro avg', 'weighted avg']] = report_custom_df[['precision', 'recall', 'f1-score', 'accuracy', 'macro avg', 'weighted avg']].round(2)
            report_custom_df['support'] = report_custom_df['support'].astype(int)
            st.dataframe(report_custom_df, use_container_width=True)

            # Confusion Matrix for Threshold=0.5
            st.subheader("Confusion Matrix (Threshold=0.5)")
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix (Threshold=0.5)',
                            labels=dict(x='Predicted', y='Actual'),
                            x=['Not Churned', 'Churned'], y=['Not Churned', 'Churned'])
            fig.update_layout(width=700, height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Confusion Matrix for Threshold=0.3
            st.subheader("Confusion Matrix (Threshold=0.3)")
            cm_custom = confusion_matrix(y_test, y_pred_custom)
            fig = px.imshow(cm_custom, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix (Threshold=0.3)',
                            labels=dict(x='Predicted', y='Actual'),
                            x=['Not Churned', 'Churned'], y=['Not Churned', 'Churned'])
            fig.update_layout(width=700, height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Feature Importance (SHAP) - XGBoost Classifier")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            plt.figure(figsize=(4, 2.5))
            shap.summary_plot(shap_values, X_test, feature_names=X.columns, plot_type='bar', show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())

            # Churn Prediction Form
            st.subheader("Predict Churn for New Customer (XGBoost Classifier)")
            with st.form("churn_prediction_form"):
                input_data = {}
                numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns.drop(['Churn'])
                categorical_cols = df_clean.select_dtypes(include=['object']).columns
                col1, col2 = st.columns(2)
                with col1:
                    for col in numerical_cols:
                        min_val, max_val = df_clean[col].min(), df_clean[col].max()
                        input_data[col] = st.slider(f"{col}", float(min_val), float(max_val), float(df_clean[col].median()), step=0.1)
                with col2:
                    for col in categorical_cols:
                        options = df_clean[col].unique().tolist()
                        input_data[col] = st.selectbox(f"{col}", options)
                input_data['AvgOrderValue'] = input_data['CashbackAmount'] / max(input_data['OrderCount'], 1)
                input_data['RecentComplaint'] = 1 if input_data['Complain'] == 1 else 0
                submitted = st.form_submit_button("Predict Churn")
                if submitted:
                    input_df = pd.DataFrame([input_data])
                    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
                    missing_cols = set(X.columns) - set(input_df.columns)
                    for col in missing_cols:
                        input_df[col] = 0
                    input_df = input_df[X.columns]
                    input_scaled = scaler.transform(input_df)
                    churn_prob = model.predict_proba(input_scaled)[0, 1]
                    churn_pred = 1 if churn_prob >= 0.3 else 0
                    st.write("### Prediction Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Churn Probability**: {churn_prob:.2%}")
                        st.write(f"**Prediction (Churn if Probability ≥ 0.3)**: {'Churn' if churn_pred == 1 else 'No Churn'}")
                        st.markdown("*Note: A threshold of 0.3 is used to prioritize identifying at-risk customers, given the imbalanced dataset.*")
                    with col2:
                        st.write("**Input Data (After Encoding)**:")
                        st.dataframe(input_df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error in churn prediction: {str(e)}")
    else:
        st.error("Please upload and preprocess data on the Home page.")

# Customer Segmentation Section
if section == "Customer Segmentation":
    st.header("Customer Segmentation")
    if 'df_clean' in st.session_state and 'best_model' in st.session_state and 'X_columns' in st.session_state:
        try:
            df_clean = st.session_state['df_clean']
            best_model = st.session_state['best_model']
            X_columns = st.session_state['X_columns']
            scaler = st.session_state['scaler']
            df_rfm = df_clean[['DaySinceLastOrder', 'OrderCount', 'CashbackAmount']].copy()
            df_rfm.columns = ['Recency', 'Frequency', 'Monetary']
            scaler_rfm = StandardScaler()
            rfm_scaled = scaler_rfm.fit_transform(df_rfm)
            best_k = 4
            final_kmeans = KMeans(n_clusters=best_k, random_state=42)
            df_rfm['KMeans_Cluster'] = final_kmeans.fit_predict(rfm_scaled)
            cluster_means = df_rfm.groupby('KMeans_Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
            best_cluster = cluster_means['Monetary'].idxmax()
            df_rfm['Segment_Label'] = df_rfm['KMeans_Cluster'].apply(lambda x: 'High Value' if x == best_cluster else 'Low Value')
            def subgroup_label(row):
                try:
                    if row['Recency'] <= df_rfm['Recency'].quantile(0.25) and \
                       row['Frequency'] >= df_rfm['Frequency'].quantile(0.75) and \
                       row['Monetary'] >= df_rfm['Monetary'].quantile(0.75):
                        return 'Champion'
                    elif row['Frequency'] >= df_rfm['Frequency'].quantile(0.75):
                        return 'Loyal'
                    elif row['Monetary'] >= df_rfm['Monetary'].quantile(0.75):
                        return 'Big Spender'
                    elif row['Recency'] >= df_rfm['Recency'].quantile(0.75):
                        return 'At Risk'
                    elif row['Recency'] <= df_rfm['Recency'].quantile(0.25):
                        return 'New'
                    else:
                        return 'Mid-Value'
                except:
                    return 'Mid-Value'
            df_rfm['RFM_Subgroup'] = df_rfm.apply(subgroup_label, axis=1)
            # Compute churn probabilities
            X = df_clean.drop(columns=['Churn'])
            X = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=True)
            X = X[X_columns]
            X_scaled = scaler.transform(X)
            df_rfm['Churn_Probability'] = best_model.predict_proba(X_scaled)[:, 1]
            st.session_state['df_rfm'] = df_rfm
            st.subheader("Segment Distribution")
            fig = px.histogram(df_rfm, x='Segment_Label', title='Customer Segment Distribution')
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("RFM Subgroup Distribution")
            fig = px.histogram(df_rfm, x='RFM_Subgroup', title='RFM Subgroup Distribution')
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Churn Probability by RFM Subgroup")
            fig = px.box(df_rfm, x='RFM_Subgroup', y='Churn_Probability', title='Churn Probability by RFM Subgroup')
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("High-Risk Customers (Churn > 0.7)")
            high_risk = df_rfm[df_rfm['Churn_Probability'] > 0.7]
            high_risk_counts = high_risk['RFM_Subgroup'].value_counts()
            fig = px.bar(x=high_risk_counts.index, y=high_risk_counts.values, title='High-Risk Customers by RFM Subgroup')
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in segmentation: {str(e)}")
    else:
        st.error("Please complete previous steps.")

# Segment Summary Section
if section == "Segment Summary":
    st.header("Segment Summary")
    if 'df_rfm' in st.session_state:
        try:
            df_rfm = st.session_state['df_rfm']
            st.subheader("Segment Summary")
            segment_summary = df_rfm.groupby('Segment_Label').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'Churn_Probability': 'mean'
            }).round(2)
            st.write(segment_summary)
            st.subheader("RFM Subgroup Summary")
            RFM_subgroup_summary = df_rfm.groupby('RFM_Subgroup').agg({
                'Recency': 'mean',
                'Frequency': 'mean',
                'Monetary': 'mean',
                'Churn_Probability': 'mean'
            }).round(2)
            st.write(RFM_subgroup_summary)
            csv = segment_summary.to_csv(index=True)
            st.download_button("Download Segment Summary", csv, "segment_summary.csv", "text/csv")
        except Exception as e:
            st.error(f"Error in segment summary: {str(e)}")
    else:
        st.error("Please complete the segmentation step.")

# Targeted Treatment Section
if section == "Targeted Treatment":
    st.header("Targeted Treatment")
    if 'df_rfm' in st.session_state:
        try:
            df_rfm = st.session_state['df_rfm']
            # Define treatment strategies for each RFM subgroup
            treatment_strategies = {
                'Champion': {
                    'Description': 'Recent, frequent, high-spending customers with low churn risk.',
                    'Recommended Treatment': 'Offer VIP rewards, exclusive product previews, or loyalty program benefits to maintain engagement.',
                    'Priority': 'Low'
                },
                'Loyal': {
                    'Description': 'Frequent buyers with moderate recency and spending, low to medium churn risk.',
                    'Recommended Treatment': 'Provide loyalty discounts, cross-sell complementary products, or invite to special events.',
                    'Priority': 'Medium'
                },
                'Big Spender': {
                    'Description': 'High-spending customers with moderate frequency and recency, low to medium churn risk.',
                    'Recommended Treatment': 'Offer personalized product recommendations, premium support, or bundle deals to encourage repeat purchases.',
                    'Priority': 'Medium'
                },
                'At Risk': {
                    'Description': 'Inactive customers with high recency and high churn risk.',
                    'Recommended Treatment': 'Send re-engagement emails with time-limited discounts (e.g., 20% off next purchase) or personalized reminders.',
                    'Priority': 'High'
                },
                'New': {
                    'Description': 'Recent customers with low frequency and spending, medium churn risk.',
                    'Recommended Treatment': 'Send welcome emails, offer onboarding discounts (e.g., 10% off first order), or guide through product discovery.',
                    'Priority': 'Medium'
                },
                'Mid-Value': {
                    'Description': 'Average customers with balanced recency, frequency, and spending, medium churn risk.',
                    'Recommended Treatment': 'Target with seasonal promotions, upsell opportunities, or feedback surveys to boost engagement.',
                    'Priority': 'Medium'
                }
            }
            # Calculate mean churn probability per subgroup
            churn_prob_means = df_rfm.groupby('RFM_Subgroup')['Churn_Probability'].mean().round(2).to_dict()
            # Create treatment DataFrame
            treatment_df = pd.DataFrame([
                {
                    'RFM Subgroup': subgroup,
                    'Description': details['Description'],
                    'Churn Probability (Mean)': churn_prob_means.get(subgroup, 0.0),
                    'Recommended Treatment': details['Recommended Treatment'],
                    'Priority': details['Priority']
                }
                for subgroup, details in treatment_strategies.items()
                if subgroup in df_rfm['RFM_Subgroup'].unique()
            ])
            # Display treatment summary table
            st.subheader("Treatment Strategies by RFM Subgroup")
            st.write("The table below outlines recommended actions to retain or engage customers based on their RFM subgroup.")
            st.dataframe(treatment_df, use_container_width=True)
            # Download button for treatment strategies
            csv = treatment_df.to_csv(index=False)
            st.download_button("Download Treatment Strategies", csv, "treatment_strategies.csv", "text/csv")
            # Interactive treatment details
            st.subheader("Detailed Treatment Recommendations")
            selected_subgroup = st.selectbox("Select RFM Subgroup", treatment_df['RFM Subgroup'].unique())
            selected_details = treatment_df[treatment_df['RFM Subgroup'] == selected_subgroup].iloc[0]
            st.write(f"**{selected_subgroup}**")
            st.write(f"- **Description**: {selected_details['Description']}")
            st.write(f"- **Churn Probability (Mean)**: {selected_details['Churn Probability (Mean)']:.2f}")
            st.write(f"- **Recommended Treatment**: {selected_details['Recommended Treatment']}")
            st.write(f"- **Priority**: {selected_details['Priority']}")
            # Visualization: Customer count by subgroup with priority coloring
            st.subheader("Customer Distribution by Priority")
            subgroup_counts = df_rfm['RFM_Subgroup'].value_counts().reset_index()
            subgroup_counts.columns = ['RFM Subgroup', 'Customer Count']
            subgroup_counts = subgroup_counts.merge(treatment_df[['RFM Subgroup', 'Priority']], on='RFM Subgroup')
            priority_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            subgroup_counts['Color'] = subgroup_counts['Priority'].map(priority_colors)
            fig = px.bar(subgroup_counts, x='RFM Subgroup', y='Customer Count', color='Priority',
                         color_discrete_map=priority_colors, title='Customer Count by RFM Subgroup and Priority')
            fig.update_layout(width=600, height=400)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in targeted treatment: {str(e)}")
    else:
        st.error("Please complete the segmentation step.")