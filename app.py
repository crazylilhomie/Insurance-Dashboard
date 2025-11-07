import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="Insurance Policy Dashboard",
    page_icon="🛡️",
    layout="wide"
)

# --- Data Loading and Caching ---
@st.cache_data
def load_data(filepath):
    """
    Loads and preprocesses the insurance data.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Error loading Insurance.csv: {e}")
        st.error("Ensure 'Insurance.csv' is in the same directory as 'app.py'.")
        return pd.DataFrame(), None

    # Drop identifier columns
    df = df.drop(columns=['POLICY_NO', 'PI_NAME'], errors='ignore')

    # Clean numeric columns with commas/quotes
    for col in ['SUM_ASSURED', 'PI_ANNUAL_INCOME']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include='object').columns.drop('POLICY_STATUS', errors='ignore')
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')

    df = df.replace(r'^\s*$', 'Unknown', regex=True)

    # Encode target variable
    if 'POLICY_STATUS' in df.columns:
        le = LabelEncoder()
        df['POLICY_STATUS_ENCODED'] = le.fit_transform(df['POLICY_STATUS'])
        st.session_state['label_encoder_classes'] = le.classes_
        return df, le
    else:
        st.error("Target column 'POLICY_STATUS' not found in dataset.")
        return pd.DataFrame(), None

# Load data
df, label_encoder = load_data("Insurance.csv")

if not df.empty:
    st.title("🛡️ Insurance Policy Analysis and Prediction Dashboard")
    tab1, tab2, tab3 = st.tabs(
        ["📈 Insurance Insights Dashboard", "🤖 Model Performance", "🔮 Make New Predictions"]
    )

    # =================== TAB 1 ===================
    with tab1:
        st.header("Insurance Insights Dashboard")
        st.sidebar.header("Dashboard Filters")

        # Filters
        selected_occupations = df['PI_OCCUPATION'].unique().tolist() if 'PI_OCCUPATION' in df.columns else []
        selected_occupations = st.sidebar.multiselect("Filter by Job Role", options=selected_occupations, default=selected_occupations)
        min_age, max_age = int(df['PI_AGE'].min()), int(df['PI_AGE'].max()) if 'PI_AGE' in df.columns else (0,100)
        selected_age_range = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))

        # Filter dataframe
        filtered_df = df.copy()
        if 'PI_OCCUPATION' in df.columns:
            filtered_df = filtered_df[filtered_df['PI_OCCUPATION'].isin(selected_occupations)]
        if 'PI_AGE' in df.columns:
            filtered_df = filtered_df[(filtered_df['PI_AGE'] >= selected_age_range[0]) & (filtered_df['PI_AGE'] <= selected_age_range[1])]

        if filtered_df.empty:
            st.warning("No data matches filters.")
        else:
            # KPI metrics
            total_claims = filtered_df.shape[0]
            approved_claims = filtered_df[filtered_df['POLICY_STATUS']=='Approved Death Claim'].shape[0]
            approval_rate = (approved_claims/total_claims)*100 if total_claims>0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Claims", f"{total_claims:,}")
            col2.metric("Approved Claims", f"{approved_claims:,}")
            col3.metric("Approval Rate", f"{approval_rate:.2f}%")

            # Charts
            col1, col2 = st.columns(2)

            # Chart 1: Policy Status by Occupation
            if 'PI_OCCUPATION' in filtered_df.columns:
                df_grouped = filtered_df.groupby(['PI_OCCUPATION','POLICY_STATUS']).size().reset_index(name='Count')
                fig1 = px.bar(df_grouped.nlargest(20,'Count'), x='PI_OCCUPATION', y='Count', color='POLICY_STATUS',
                              barmode='group', title='Top 20 Occupations by Policy Status',
                              color_discrete_map={'Approved Death Claim':'green','Repudiate Death':'red'})
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)

            # Chart 2: Sum Assured vs Income
            if 'PI_ANNUAL_INCOME' in filtered_df.columns and 'SUM_ASSURED' in filtered_df.columns:
                fig2 = px.scatter(filtered_df, x='PI_ANNUAL_INCOME', y='SUM_ASSURED', color='POLICY_STATUS',
                                  title='Sum Assured vs Annual Income', opacity=0.7,
                                  color_discrete_map={'Approved Death Claim':'green','Repudiate Death':'red'},
                                  hover_data=['PI_AGE','PI_OCCUPATION'])
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)

            # Chart 3: Age distribution
            if 'PI_AGE' in filtered_df.columns:
                fig3 = px.violin(filtered_df, y='PI_AGE', x='POLICY_STATUS', color='POLICY_STATUS',
                                 box=True, points='all', title='Age Distribution by Policy Status',
                                 color_discrete_map={'Approved Death Claim':'green','Repudiate Death':'red'})
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)

            # Chart 4: Claim Reason & Timing
            if 'REASON_FOR_CLAIM' in filtered_df.columns and 'EARLY_NON' in filtered_df.columns:
                fig4 = px.sunburst(filtered_df.loc[filtered_df['REASON_FOR_CLAIM']!='Unknown'],
                                   path=['POLICY_STATUS','EARLY_NON','REASON_FOR_CLAIM'],
                                   title='Claims: Status -> Timing -> Reason', color_discrete_sequence=px.colors.qualitative.Pastel)
                fig4.update_layout(height=400)
                st.plotly_chart(fig4, use_container_width=True)

            # Chart 5: Geographic Sum Assured
            if 'PI_STATE' in filtered_df.columns and 'SUM_ASSURED' in filtered_df.columns:
                df_state = filtered_df.groupby(['PI_STATE','POLICY_STATUS'])['SUM_ASSURED'].sum().reset_index()
                fig5 = px.treemap(df_state, path=['PI_STATE','POLICY_STATUS'], values='SUM_ASSURED',
                                  title='Sum Assured by State & Status', color='SUM_ASSURED', color_continuous_scale='Reds')
                fig5.update_layout(height=820)
                st.plotly_chart(fig5, use_container_width=True)

    # =================== TAB 2 ===================
    with tab2:
        st.header("🤖 Model Performance")
        st.info("Models are trained on the full preprocessed dataset using 5-fold CV.")

        if st.button("Run Classification Models"):
            X = df.drop(columns=['POLICY_STATUS','POLICY_STATUS_ENCODED'])
            y = df['POLICY_STATUS_ENCODED']
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include='object').columns.tolist()

            # Preprocessing
            numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler',StandardScaler())])
            categorical_transformer = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='Unknown')),
                                                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            preprocessor = ColumnTransformer([('num',numeric_transformer,numeric_features),
                                             ('cat',categorical_transformer,categorical_features)],
                                             remainder='passthrough')

            # Models
            models = {"Decision Tree": DecisionTreeClassifier(random_state=42),
                      "Random Forest": RandomForestClassifier(random_state=42),
                      "Gradient Boosted Tree": GradientBoostingClassifier(random_state=42)}

            scoring_metrics = ['accuracy','precision_weighted','recall_weighted','f1_weighted','roc_auc']
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            metrics_results = {}

            for name, model in models.items():
                pipeline = Pipeline([('preprocessor',preprocessor),('classifier',model)])
                cv_results = cross_validate(pipeline,X,y,cv=kfold,scoring=scoring_metrics,return_train_score=True)
                metrics_results[name] = {'Train Accuracy': np.mean(cv_results['train_accuracy']),
                                         'Test Accuracy': np.mean(cv_results['test_accuracy']),
                                         'Precision': np.mean(cv_results['test_precision_weighted']),
                                         'Recall': np.mean(cv_results['test_recall_weighted']),
                                         'F1 Score': np.mean(cv_results['test_f1_weighted']),
                                         'AUC ROC': np.mean(cv_results['test_roc_auc'])}
            st.dataframe(pd.DataFrame(metrics_results).T.style.highlight_max(axis=0, color='lightgreen'))

    # =================== TAB 3 ===================
    with tab3:
        st.header("🔮 Predict Policy Status for New Data")
        @st.cache_resource
        def get_prediction_model():
            X = df.drop(columns=['POLICY_STATUS','POLICY_STATUS_ENCODED'])
            y = df['POLICY_STATUS_ENCODED']
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include='object').columns.tolist()
            numeric_transformer = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
            categorical_transformer = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='Unknown')),
                                                ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])
            preprocessor = ColumnTransformer([('num',numeric_transformer,numeric_features),
                                             ('cat',categorical_transformer,categorical_features)],
                                             remainder='passthrough')
            final_pipeline = Pipeline([('preprocessor',preprocessor),('classifier',RandomForestClassifier(random_state=42))])
            final_pipeline.fit(X,y)
            return final_pipeline

        prediction_pipeline = get_prediction_model()
        uploaded_file = st.file_uploader("Upload CSV for prediction", type=["csv"])
        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)
            for col in ['SUM_ASSURED','PI_ANNUAL_INCOME']:
                if col in new_data.columns:
                    new_data[col] = new_data[col].astype(str).str.replace('"','').str.replace(',','')
                    new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
            preds_encoded = prediction_pipeline.predict(new_data)
            preds_proba = prediction_pipeline.predict_proba(new_data)
            le_classes = st.session_state.get('label_encoder_classes',['Approved Death Claim','Repudiate Death'])
            preds_decoded = np.array(le_classes)[preds_encoded]
            new_data['PREDICTED_POLICY_STATUS'] = preds_decoded
            try:
                approved_class_index = np.where(np.array(le_classes)=='Approved Death Claim')[0][0]
                new_data['PROBABILITY_APPROVED'] = preds_proba[:,approved_class_index]
            except:
                new_data['PROBABILITY_CLASS_1'] = preds_proba[:,1]

            st.subheader("Prediction Results")
            st.dataframe(new_data)

            @st.cache_data
            def convert_df_to_csv(df_to_convert):
                return df_to_convert.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions", data=convert_df_to_csv(new_data), file_name="policy_predictions.csv", mime="text/csv")

else:
    st.error("Failed to load 'Insurance.csv'. Please ensure it is present and valid.")
