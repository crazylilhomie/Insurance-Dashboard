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
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        st.error(f"Error loading Insurance.csv: {e}")
        st.error("Ensure 'Insurance.csv' is in the same directory as 'app.py'.")
        return pd.DataFrame(), None

    # Drop identifier columns
    df = df.drop(columns=['POLICY_NO', 'PI_NAME'], errors='ignore')

    # Clean numeric columns
    for col in ['SUM_ASSURED', 'PI_ANNUAL_INCOME']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns.drop('POLICY_STATUS', errors='ignore'):
        df[col] = df[col].fillna('Unknown')
    df = df.replace(r'^\s*$', 'Unknown', regex=True)

    # Encode target
    if 'POLICY_STATUS' in df.columns:
        le = LabelEncoder()
        df['POLICY_STATUS_ENCODED'] = le.fit_transform(df['POLICY_STATUS'])
        st.session_state['label_encoder_classes'] = le.classes_
        return df, le
    else:
        st.error("Target column 'POLICY_STATUS' not found.")
        return pd.DataFrame(), None

# Load data
df, label_encoder = load_data("Insurance.csv")

if not df.empty:
    st.title("🛡️ Insurance Policy Analysis and Prediction Dashboard")

    tab1, tab2, tab3 = st.tabs(
        ["📈 Insurance Insights Dashboard", "🤖 Model Performance", "🔮 Make New Predictions"]
    )

    # ================== TAB 1 ==================
    with tab1:
        st.header("Insurance Insights Dashboard")
        st.sidebar.header("Dashboard Filters")

        # Filters
        selected_occupations = sorted(df['PI_OCCUPATION'].unique()) if 'PI_OCCUPATION' in df.columns else []
        if 'PI_OCCUPATION' in df.columns:
            selected_occupations = st.sidebar.multiselect("Filter by Job Role", options=selected_occupations, default=selected_occupations)

        min_age, max_age = (0, 100)
        if 'PI_AGE' in df.columns:
            min_age, max_age = int(df['PI_AGE'].min()), int(df['PI_AGE'].max())
            selected_age_range = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        else:
            selected_age_range = (0, 100)

        # Filtered Data
        filtered_df = df.copy()
        if 'PI_OCCUPATION' in df.columns:
            filtered_df = filtered_df[filtered_df['PI_OCCUPATION'].isin(selected_occupations)]
        if 'PI_AGE' in df.columns:
            filtered_df = filtered_df[(filtered_df['PI_AGE']>=selected_age_range[0]) & (filtered_df['PI_AGE']<=selected_age_range[1])]

        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
        else:
            st.subheader("Key Performance Indicators")
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            total_claims = filtered_df.shape[0]
            approved_claims = filtered_df[filtered_df['POLICY_STATUS']=='Approved Death Claim'].shape[0]
            approval_rate = (approved_claims/total_claims)*100 if total_claims>0 else 0
            col_kpi1.metric("Total Claims", f"{total_claims:,}")
            col_kpi2.metric("Approved Claims", f"{approved_claims:,}")
            col_kpi3.metric("Approval Rate", f"{approval_rate:.2f}%")

            st.subheader("Visual Analysis")
            col1, col2 = st.columns(2)
            with col1:
                # Policy Status by Occupation
                if 'PI_OCCUPATION' in filtered_df.columns:
                    df_grouped = filtered_df.groupby(['PI_OCCUPATION','POLICY_STATUS']).size().reset_index(name='Count')
                    fig1 = px.bar(df_grouped.nlargest(20,'Count'), x='PI_OCCUPATION', y='Count', color='POLICY_STATUS', barmode='group',
                                  color_discrete_map={'Approved Death Claim':'green','Repudiate Death':'red'},
                                  title='Top 20 Occupations by Policy Status')
                    st.plotly_chart(fig1, use_container_width=True)

                # Sum Assured vs Annual Income
                if 'PI_ANNUAL_INCOME' in filtered_df.columns and 'SUM_ASSURED' in filtered_df.columns:
                    fig2 = px.scatter(filtered_df, x='PI_ANNUAL_INCOME', y='SUM_ASSURED', color='POLICY_STATUS',
                                      opacity=0.7, hover_data=['PI_AGE','PI_OCCUPATION'],
                                      color_discrete_map={'Approved Death Claim':'green','Repudiate Death':'red'},
                                      title='Sum Assured vs Annual Income by Status')
                    st.plotly_chart(fig2, use_container_width=True)

                # Age Distribution
                if 'PI_AGE' in filtered_df.columns:
                    fig3 = px.violin(filtered_df, x='POLICY_STATUS', y='PI_AGE', color='POLICY_STATUS', box=True, points='all',
                                     color_discrete_map={'Approved Death Claim':'green','Repudiate Death':'red'},
                                     title='Age Distribution by Policy Status')
                    st.plotly_chart(fig3, use_container_width=True)

            with col2:
                # Claim Reason Sunburst
                if 'REASON_FOR_CLAIM' in filtered_df.columns and 'EARLY_NON' in filtered_df.columns:
                    fig4 = px.sunburst(filtered_df.loc[filtered_df['REASON_FOR_CLAIM']!='Unknown'],
                                       path=['POLICY_STATUS','EARLY_NON','REASON_FOR_CLAIM'],
                                       color_discrete_sequence=px.colors.qualitative.Pastel,
                                       title='Sunburst of Claims: Status -> Timing -> Reason')
                    st.plotly_chart(fig4, use_container_width=True)

                # Sum Assured by State
                if 'PI_STATE' in filtered_df.columns and 'SUM_ASSURED' in filtered_df.columns:
                    df_state = filtered_df.groupby(['PI_STATE','POLICY_STATUS'])['SUM_ASSURED'].sum().reset_index()
                    fig5 = px.treemap(df_state, path=['PI_STATE','POLICY_STATUS'], values='SUM_ASSURED', color='SUM_ASSURED',
                                      color_continuous_scale='Reds', title='Total Sum Assured by State & Status')
                    st.plotly_chart(fig5, use_container_width=True)

    # ================== TAB 2 ==================
    with tab2:
        st.header("🤖 Model Performance")
        st.info("Models are trained on the full dataset with 5-fold CV.")

        if st.button("Run Classification Models"):
            X = df.drop(columns=['POLICY_STATUS','POLICY_STATUS_ENCODED'])
            y = df['POLICY_STATUS_ENCODED']
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include='object').columns.tolist()

            numeric_transformer = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
            categorical_transformer = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='Unknown')),
                                                ('onehot',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            preprocessor = ColumnTransformer([('num',numeric_transformer,numeric_features),
                                             ('cat',categorical_transformer,categorical_features)],
                                             remainder='passthrough')

            models = {"Decision Tree": DecisionTreeClassifier(random_state=42),
                      "Random Forest": RandomForestClassifier(random_state=42),
                      "Gradient Boosted Tree": GradientBoostingClassifier(random_state=42)}

            # Cross-validation metrics
            scoring_metrics = ['accuracy','precision_weighted','recall_weighted','f1_weighted','roc_auc']
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            metrics_results = {}
            st.subheader("Cross-Validation Metrics")
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

            # Confusion matrix & ROC
            st.subheader("Confusion Matrix & ROC Curve (Single Train-Test Split)")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            plot_cols = st.columns(len(models))
            for i, (name, model) in enumerate(models.items()):
                with plot_cols[i]:
                    st.markdown(f"**{name}**")
                    pipeline = Pipeline([('preprocessor',preprocessor),('classifier',model)])
                    pipeline.fit(X_train,y_train)
                    y_pred = pipeline.predict(X_test)
                    y_proba = pipeline.predict_proba(X_test) if hasattr(model,'predict_proba') else pipeline.decision_function(X_test)
                    if y_proba.ndim>1 and y_proba.shape[1]>1:
                        y_proba_plot = y_proba[:,1]
                    else:
                        y_proba_plot = y_proba

                    target_names = st.session_state.get('label_encoder_classes',['Class 0','Class 1'])
                    fig_cm, ax_cm = plt.subplots(figsize=(4,4))
                    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=target_names, cmap='Blues', ax=ax_cm)
                    st.pyplot(fig_cm)

                    fig_roc, ax_roc = plt.subplots(figsize=(4,4))
                    RocCurveDisplay.from_predictions(y_test, y_proba_plot, ax=ax_roc)
                    st.pyplot(fig_roc)

    # ================== TAB 3 ==================
    with tab3:
        st.header("🔮 Predict Policy Status for New Data")

        @st.cache_resource
        def get_prediction_model():
            X_all = df.drop(columns=['POLICY_STATUS','POLICY_STATUS_ENCODED'])
            y_all = df['POLICY_STATUS_ENCODED']
            numeric_features = X_all.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_all.select_dtypes(include='object').columns.tolist()
            numeric_transformer = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
            categorical_transformer = Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='Unknown')),
                                                ('onehot',OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
            preprocessor = ColumnTransformer([('num',numeric_transformer,numeric_features),
                                             ('cat',categorical_transformer,categorical_features)],
                                             remainder='passthrough')
            final_pipeline = Pipeline([('preprocessor',preprocessor),('classifier',RandomForestClassifier(random_state=42))])
            final_pipeline.fit(X_all,y_all)
            return final_pipeline

        prediction_pipeline = get_prediction_model()
        uploaded_file = st.file_uploader("Upload your CSV", type=['csv'])
        if uploaded_file is not None:
            new_data = pd.read_csv(uploaded_file)
            for col in ['SUM_ASSURED','PI_ANNUAL_INCOME']:
                if col in new_data.columns:
                    new_data[col] = new_data[col].astype(str).str.replace('"','').str.replace(',','')
                    new_data[col] = pd.to_numeric(new_data[col],errors='coerce')
            predictions_encoded = prediction_pipeline.predict(new_data)
            predictions_proba = prediction_pipeline.predict_proba(new_data)
            le_classes = st.session_state.get('label_encoder_classes',['Approved Death Claim','Repudiate Death'])
            predictions_decoded = le_classes[predictions_encoded]
            new_data['PREDICTED_POLICY_STATUS'] = predictions_decoded
            try:
                approved_index = np.where(le_classes=='Approved Death Claim')[0][0]
                new_data['PROBABILITY_APPROVED'] = predictions_proba[:,approved_index]
            except:
                new_data['PROBABILITY_CLASS_1'] = predictions_proba[:,1]
            st.dataframe(new_data)
            st.download_button("Download CSV", new_data.to_csv(index=False).encode('utf-8'), "predictions.csv", "text/csv")
else:
    st.error("Failed to load 'Insurance.csv'. Ensure the file exists and is valid.")
