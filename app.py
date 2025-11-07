import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import io

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
        st.error("Please make sure 'Insurance.csv' is in the same directory as 'app.py' in your GitHub repository.")
        return pd.DataFrame(), None

    # Drop identifier columns
    df = df.drop(columns=['POLICY_NO', 'PI_NAME'], errors='ignore')

    # Clean numeric columns with commas and quotes
    for col in ['SUM_ASSURED', 'PI_ANNUAL_INCOME']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('"', '').str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs
    # Numerical columns: fill with median
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical columns: fill with 'Unknown'
    cat_cols = df.select_dtypes(include='object').columns.drop('POLICY_STATUS', errors='ignore')
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        
    # Handle potential empty strings
    df = df.replace(r'^\s*$', 'Unknown', regex=True)

    # Encode the target variable
    if 'POLICY_STATUS' in df.columns:
        le = LabelEncoder()
        df['POLICY_STATUS_ENCODED'] = le.fit_transform(df['POLICY_STATUS'])
        # Store classes in session state for later use in prediction
        st.session_state['label_encoder_classes'] = le.classes_
        return df, le
    else:
        st.error("Target column 'POLICY_STATUS' not found in the dataset.")
        return pd.DataFrame(), None

# Load the data
df, label_encoder = load_data("Insurance.csv")

if not df.empty:
    # --- App Title ---
    st.title("🛡️ Insurance Policy Analysis and Prediction Dashboard")

    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(
        ["📈 Insurance Insights Dashboard", "🤖 Model Performance", "🔮 Make New Predictions"]
    )

    # ==============================================================================
    # --- TAB 1: INSURANCE INSIGHTS DASHBOARD ---
    # ==============================================================================
    with tab1:
        st.header("Insurance Insights Dashboard")
        st.write("Analyze policyholder demographics, claim reasons, and policy status.")

        # --- Sidebar Filters ---
        st.sidebar.header("Dashboard Filters")

        # Job Role Filter (Multiselect)
        if 'PI_OCCUPATION' in df.columns:
            occupations = sorted(df['PI_OCCUPATION'].unique())
            selected_occupations = st.sidebar.multiselect(
                "Filter by Job Role",
                options=occupations,
                default=occupations
            )
        else:
            selected_occupations = []
            st.sidebar.warning("Column 'PI_OCCUPATION' not found.")

        # Age Filter (Slider)
        if 'PI_AGE' in df.columns:
            min_age, max_age = int(df['PI_AGE'].min()), int(df['PI_AGE'].max())
            selected_age_range = st.sidebar.slider(
                "Filter by Policyholder Age",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age)
            )
        else:
            selected_age_range = (0, 100)
            st.sidebar.warning("Column 'PI_AGE' not found.")

        # --- Filter Data ---
        filtered_df = df.copy()
        if 'PI_OCCUPATION' in df.columns:
            filtered_df = filtered_df[filtered_df['PI_OCCUPATION'].isin(selected_occupations)]
        if 'PI_AGE' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['PI_AGE'] >= selected_age_range[0]) &
                (filtered_df['PI_AGE'] <= selected_age_range[1])
            ]

        if filtered_df.empty:
            st.warning("No data matches the selected filters. Please adjust your filter settings.")
        else:
            # --- Dashboard Charts ---
            st.subheader("Key Performance Indicators")
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            
            total_claims = filtered_df.shape[0]
            approved_claims = filtered_df[filtered_df['POLICY_STATUS'] == 'Approved Death Claim'].shape[0]
            repudiated_claims = total_claims - approved_claims
            approval_rate = (approved_claims / total_claims) * 100 if total_claims > 0 else 0
            
            col_kpi1.metric("Total Claims in View", f"{total_claims:,}")
            col_kpi2.metric("Approved Claims", f"{approved_claims:,}")
            col_kpi3.metric("Approval Rate", f"{approval_rate:.2f}%")

            st.subheader("Visual Analysis")
            
            # Layout columns
            col1, col2 = st.columns(2)

            with col1:
                # Chart 1: Policy Status by Occupation
                if 'PI_OCCUPATION' in filtered_df.columns:
                    st.write("#### Policy Status by Occupation")
                    df_grouped = filtered_df.groupby(['PI_OCCUPATION', 'POLICY_STATUS']).size().reset_index(name='Count')
                    fig1 = px.bar(
                        df_grouped.nlargest(20, 'Count'), # Show top 20 occupations
                        x='PI_OCCUPATION',
                        y='Count',
                        color='POLICY_STATUS',
                        barmode='group',
                        title='Top 20 Occupations by Policy Status',
                        color_discrete_map={'Approved Death Claim': 'green', 'Repudiate Death': 'red'}
                    )
                    fig1.update_layout(minheight=400)
                    st.plotly_chart(fig1, use_container_width=True)
                
                # Chart 2: Sum Assured vs. Annual Income by Policy Status
                if 'PI_ANNUAL_INCOME' in filtered_df.columns and 'SUM_ASSURED' in filtered_df.columns:
                    st.write("#### Sum Assured vs. Annual Income")
                    fig2 = px.scatter(
                        filtered_df,
                        x='PI_ANNUAL_INCOME',
                        y='SUM_ASSURED',
                        color='POLICY_STATUS',
                        title='Sum Assured vs. Annual Income by Policy Status',
                        opacity=0.7,
                        color_discrete_map={'Approved Death Claim': 'green', 'Repudiate Death': 'red'},
                        hover_data=['PI_AGE', 'PI_OCCUPATION']
                    )
                    fig2.update_layout(minheight=400)
                    st.plotly_chart(fig2, use_container_width=True)

                # Chart 3: Age Distribution by Policy Status
                if 'PI_AGE' in filtered_df.columns:
                    st.write("#### Age Distribution by Policy Status")
                    fig3 = px.violin(
                        filtered_df,
                        y='PI_AGE',
                        x='POLICY_STATUS',
                        color='POLICY_STATUS',
                        box=True,
                        points="all",
                        title='Age Distribution by Policy Status',
                        color_discrete_map={'Approved Death Claim': 'green', 'Repudiate Death': 'red'}
                    )
                    fig3.update_layout(minheight=400)
                    st.plotly_chart(fig3, use_container_width=True)

            with col2:
                # Chart 4: Claim Analysis by Reason and Timing
                if 'REASON_FOR_CLAIM' in filtered_df.columns and 'EARLY_NON' in filtered_df.columns:
                    st.write("#### Claim Reason and Timing Analysis")
                    fig4 = px.sunburst(
                        filtered_df.loc[filtered_df['REASON_FOR_CLAIM'] != 'Unknown'], # Exclude unknown reasons for clarity
                        path=['POLICY_STATUS', 'EARLY_NON', 'REASON_FOR_CLAIM'],
                        title='Sunburst of Claims: Status -> Timing -> Reason',
                        color_discrete_sequence=px.colors.qualitative.Pastel
                    )
                    fig4.update_layout(minheight=400)
                    st.plotly_chart(fig4, use_container_width=True)

                # Chart 5: Geographic Distribution of Total Sum Assured
                if 'PI_STATE' in filtered_df.columns and 'SUM_ASSURED' in filtered_df.columns:
                    st.write("#### Sum Assured by State and Policy Status")
                    df_state = filtered_df.groupby(['PI_STATE', 'POLICY_STATUS'])['SUM_ASSURED'].sum().reset_index()
                    fig5 = px.treemap(
                        df_state,
                        path=['PI_STATE', 'POLICY_STATUS'],
                        values='SUM_ASSURED',
                        title='Treemap of Total Sum Assured by State and Status',
                        color='SUM_ASSURED',
                        color_continuous_scale='Reds'
                    )
                    fig5.update_layout(minheight=820) # Taller to match other column
                    st.plotly_chart(fig5, use_container_width=True)

    # ==============================================================================
    # --- TAB 2: MODEL PERFORMANCE ---
    # ==============================================================================
    with tab2:
        st.header("🤖 Model Performance Evaluation")
        st.write("Train and evaluate classification models using 5-fold cross-validation.")
        st.info("This process may take a few moments. The models are trained on the full, preprocessed dataset.")

        if st.button("Run Classification Models"):
            with st.spinner("Processing data and training models..."):
                try:
                    # --- Data Preparation ---
                    X = df.drop(columns=['POLICY_STATUS', 'POLICY_STATUS_ENCODED'])
                    y = df['POLICY_STATUS_ENCODED']

                    # Identify numerical and categorical features
                    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                    categorical_features = X.select_dtypes(include='object').columns.tolist()

                    # --- Preprocessing Pipelines ---
                    numeric_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='median')),
                        ('scaler', StandardScaler())
                    ])

                    categorical_transformer = Pipeline(steps=[
                        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                    ])

                    # --- Column Transformer ---
                    preprocessor = ColumnTransformer(
                        transformers=[
                            ('num', numeric_transformer, numeric_features),
                            ('cat', categorical_transformer, categorical_features)
                        ],
                        remainder='passthrough'
                    )

                    # --- Models ---
                    models = {
                        "Decision Tree": DecisionTreeClassifier(random_state=42),
                        "Random Forest": RandomForestClassifier(random_state=42),
                        "Gradient Boosted Tree": GradientBoostingClassifier(random_state=42)
                    }

                    # --- Cross-Validation ---
                    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                    metrics_results = {}
                    
                    st.subheader("Cross-Validation Performance Metrics")
                    
                    scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
                    
                    for model_name, model in models.items():
                        # Create the full pipeline
                        pipeline = Pipeline(steps=[
                            ('preprocessor', preprocessor),
                            ('classifier', model)
                        ])
                        
                        # Perform cross-validation
                        cv_results = cross_validate(
                            pipeline, X, y, 
                            cv=kfold, 
                            scoring=scoring_metrics,
                            return_train_score=True
                        )
                        
                        metrics_results[model_name] = {
                            'Test Accuracy': np.mean(cv_results['test_accuracy']),
                            'Train Accuracy': np.mean(cv_results['train_accuracy']),
                            'Precision': np.mean(cv_results['test_precision_weighted']),
                            'Recall': np.mean(cv_results['test_recall_weighted']),
                            'F1 Score': np.mean(cv_results['test_f1_weighted']),
                            'AUC ROC': np.mean(cv_results['test_roc_auc'])
                        }
                    
                    # Display metrics in a DataFrame
                    metrics_df = pd.DataFrame(metrics_results).T
                    st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'))

                    # --- Plots (Confusion Matrix, Feature Importance, ROC) ---
                    st.subheader("Detailed Model Plots")
                    st.write("Models are re-trained on a single 70/30 split for plotting.")

                    # Single train-test split for plotting
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                    
                    plot_cols = st.columns(len(models))
                    
                    for i, (model_name, model) in enumerate(models.items()):
                        with plot_cols[i]:
                            st.markdown(f"**{model_name}**")
                            
                            # Create and fit pipeline
                            pipeline = Pipeline(steps=[
                                ('preprocessor', preprocessor),
                                ('classifier', model)
                            ])
                            pipeline.fit(X_train, y_train)
                            y_pred = pipeline.predict(X_test)
                            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

                            # Get target names from label encoder
                            target_names = st.session_state.get('label_encoder_classes', ['Class 0', 'Class 1'])

                            # --- Confusion Matrix ---
                            st.write("Confusion Matrix")
                            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                            ConfusionMatrixDisplay.from_predictions(
                                y_test, 
                                y_pred, 
                                display_labels=target_names, 
                                cmap='Blues', 
                                ax=ax_cm,
                                xticks_rotation='vertical'
                            )
                            st.pyplot(fig_cm)

                            # --- ROC Curve ---
                            st.write("AUC-ROC Curve")
                            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                            RocCurveDisplay.from_predictions(y_test, y_pred, ax=ax_roc)
                            st.pyplot(fig_roc)

                            # --- Feature Importance (for tree-based models) ---
                            if hasattr(model, 'feature_importances_'):
                                st.write("Top 10 Feature Importances")
                                
                                # Get feature names from preprocessor
                                try:
                                    cat_names = pipeline.named_steps['preprocessor'] \
                                                      .named_transformers_['cat'] \
                                                      .named_steps['onehot'] \
                                                      .get_feature_names_out(categorical_features)
                                    all_feature_names = numeric_features + list(cat_names)
                                    
                                    importances = pipeline.named_steps['classifier'].feature_importances_
                                    
                                    imp_df = pd.DataFrame({
                                        'Feature': all_feature_names,
                                        'Importance': importances
                                    }).sort_values(by='Importance', ascending=False).head(10)

                                    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title="Top Features")
                                    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                                    st.plotly_chart(fig_imp, use_container_width=True)
                                
                                except Exception as e:
                                    st.warning(f"Could not plot feature importance: {e}")

                except Exception as e:
                    st.error(f"An error occurred during model training: {e}")
                    st.exception(e)

    # ==============================================================================
    # --- TAB 3: MAKE NEW PREDICTIONS ---
    # ==============================================================================
    with tab3:
        st.header("🔮 Predict Policy Status for New Data")
        
        # --- Caching the prediction model ---
        @st.cache_resource
        def get_prediction_model():
            """
            Trains and caches the final prediction model (Random Forest) on all data.
            """
            st.write("Training prediction model... (This runs only once)")
            # Prepare all data
            X = df.drop(columns=['POLICY_STATUS', 'POLICY_STATUS_ENCODED'])
            y = df['POLICY_STATUS_ENCODED']

            # Define preprocessor (same as in Tab 2)
            numeric_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(include='object').columns.tolist()

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='passthrough'
            )
            
            # Create the final Random Forest pipeline
            final_pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(random_state=42))
            ])
            
            # Fit on all data
            final_pipeline.fit(X, y)
            return final_pipeline

        # Get the trained model
        try:
            prediction_pipeline = get_prediction_model()

            st.info("Upload a CSV file with the same columns as the original data (e.g., PI_GENDER, SUM_ASSURED, PI_AGE, etc.). The 'POLICY_STATUS' column is not required.")
            
            uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    new_data_processed = new_data.copy()

                    # Pre-clean the new data just like the original
                    for col in ['SUM_ASSURED', 'PI_ANNUAL_INCOME']:
                        if col in new_data_processed.columns:
                            new_data_processed[col] = new_data_processed[col].astype(str).str.replace('"', '').str.replace(',', '')
                            new_data_processed[col] = pd.to_numeric(new_data_processed[col], errors='coerce')

                    # Make predictions
                    predictions_encoded = prediction_pipeline.predict(new_data_processed)
                    predictions_proba = prediction_pipeline.predict_proba(new_data_processed)

                    # Get label names from session state
                    le_classes = st.session_state.get('label_encoder_classes', ['Approved Death Claim', 'Repudiate Death'])
                    
                    # Decode predictions
                    predictions_decoded = le_classes[predictions_encoded]
                    
                    # Add results to the dataframe
                    new_data['PREDICTED_POLICY_STATUS'] = predictions_decoded
                    
                    # Find the index of the 'Approved' class
                    try:
                        approved_class_index = np.where(le_classes == 'Approved Death Claim')[0][0]
                        new_data['PROBABILITY_APPROVED'] = predictions_proba[:, approved_class_index]
                    except Exception:
                         # Fallback if class name is different
                        new_data['PROBABILITY_CLASS_1'] = predictions_proba[:, 1]


                    st.subheader("Prediction Results")
                    st.dataframe(new_data)

                    # --- Download Button ---
                    @st.cache_data
                    def convert_df_to_csv(df_to_convert):
                        return df_to_convert.to_csv(index=False).encode('utf-8')

                    csv_output = convert_df_to_csv(new_data)
                    
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv_output,
                        file_name="policy_predictions.csv",
                        mime="text/csv",
                    )
                
                except Exception as e:
                    st.error(f"An error occurred while making predictions: {e}")
                    st.exception(e)

        except Exception as e:
            st.error(f"Failed to load the prediction model: {e}")
            st.exception(e)
else:
    st.error("The application could not start because the 'Insurance.csv' file failed to load.")
    st.info("Please ensure 'Insurance.csv' is present in your GitHub repository and is not empty or corrupted.")
