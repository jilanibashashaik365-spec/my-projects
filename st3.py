import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score, silhouette_score
)
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Classical ML Engine & Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Session State variables to store trained models
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'features_used' not in st.session_state:
    st.session_state.features_used = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'unique_vals' not in st.session_state:
    st.session_state.unique_vals = None
if 'task_trained' not in st.session_state:
    st.session_state.task_trained = None

# Title & Description
st.title("🧠 Classical ML Model Engine & Performance Dashboard")
st.markdown("""
Run **Classification**, **Regression**, and **Clustering** algorithms on datasets in real-time. 
This dashboard trains classical machine learning models and visualizes key performance metrics, decision spaces, and residuals.
""")

# Sidebar for Task & Data selection
st.sidebar.header("📁 Step 1: Select ML Task & Data")
task_type = st.sidebar.selectbox(
    "Choose Machine Learning Task",
    ["Classification", "Regression", "Clustering"]
)

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Use Demo Dataset", "Upload Custom CSV"]
)

@st.cache_data
def load_demo_data(task):
    if task == "Classification":
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['target_name'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df, iris.feature_names, 'target_name'
    
    elif task == "Regression":
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
        df['target'] = diabetes.target
        return df, diabetes.feature_names, 'target'
        
    else: # Clustering
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.80, random_state=42)
        df = pd.DataFrame(X, columns=['Annual Income (k$)', 'Spending Score (1-100)'])
        return df, ['Annual Income (k$)', 'Spending Score (1-100)'], None

# Load dataset
df = None
features = []
target_col = None

if data_source == "Use Demo Dataset":
    df, features, target_col = load_demo_data(task_type)
    st.sidebar.success(f"Loaded Demo Dataset for {task_type}")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV file successfully loaded!")
            
            all_cols = list(df.columns)
            numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

            if task_type != "Clustering":
                if task_type == "Regression":
                    target_options = numeric_cols if numeric_cols else all_cols
                else:
                    target_options = all_cols
                target_col = st.sidebar.selectbox("Select Target Variable (Y)", target_options, index=len(target_options)-1)
                feature_candidates = [c for c in numeric_cols if c != target_col]
                st.sidebar.caption(f"🔢 {len(feature_candidates)} numeric column(s) available as features.")
                features = st.sidebar.multiselect("Select Feature Variables (X)", feature_candidates, default=feature_candidates[:3])
            else:
                st.sidebar.caption(f"🔢 {len(numeric_cols)} numeric column(s) available as features.")
                features = st.sidebar.multiselect("Select Features for Clustering (at least 2)", numeric_cols, default=numeric_cols[:2])
                target_col = None
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    else:
        st.info("👋 Welcome! Please upload a CSV file or use the built-in Demo Dataset to see the ML dashboard in action.")
        st.stop()

if df is not None:
    # Display dataset info
    st.subheader("📊 Dataset Explorer")
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Total Rows", df.shape[0])
    col_stat2.metric("Total Columns", df.shape[1])
    col_stat3.metric("Selected Features count", len(features))
    
    with st.expander("🔍 View Raw Data & Summary Stats"):
        tab_preview, tab_stats = st.tabs(["Data Preview", "Summary Statistics"])
        with tab_preview:
            st.dataframe(df.head(10), use_container_width=True)
        with tab_stats:
            st.dataframe(df.describe(), use_container_width=True)

    # Preprocessing
    st.sidebar.header("⚙️ Step 2: Preprocessing")
    scale_data = st.sidebar.checkbox("Apply Standard Scaling", value=True)
    
    test_size = 0.2
    if task_type != "Clustering":
        test_size = st.sidebar.slider("Test Split Size", 0.1, 0.5, 0.2, 0.05)
    
    # Model Selection & Hyperparameters
    st.sidebar.header("🤖 Step 3: Configure ML Model")
    
    if task_type == "Classification":
        algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Gradient Boosting", "SVM", "Logistic Regression"])
        if algorithm == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, 10)
            max_depth = st.sidebar.slider("Max Depth of Trees", 2, 20, 8, 1)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif algorithm == "Gradient Boosting":
            n_estimators = st.sidebar.slider("Number of Boosting Stages", 10, 300, 100, 10)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
            max_depth = st.sidebar.slider("Max Depth of Trees", 1, 10, 3, 1)
            model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        elif algorithm == "SVM":
            c_val = st.sidebar.number_input("C (Regularization)", 0.01, 100.0, 1.0, 0.1)
            kernel = st.sidebar.selectbox("Kernel type", ["rbf", "linear", "poly"])
            model = SVC(C=c_val, kernel=kernel, probability=True, random_state=42)
        else:
            c_val = st.sidebar.number_input("C (Regularization)", 0.01, 100.0, 1.0, 0.1)
            model = LogisticRegression(C=c_val, max_iter=1000, random_state=42)
            
    elif task_type == "Regression":
        algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "Gradient Boosting", "Linear Regression", "SVM (SVR)"])
        if algorithm == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100, 10)
            max_depth = st.sidebar.slider("Max Depth of Trees", 2, 20, 8, 1)
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif algorithm == "Gradient Boosting":
            n_estimators = st.sidebar.slider("Number of Boosting Stages", 10, 300, 100, 10)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1, 0.01)
            max_depth = st.sidebar.slider("Max Depth of Trees", 1, 10, 3, 1)
            model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        elif algorithm == "Linear Regression":
            model = LinearRegression()
        else:
            c_val = st.sidebar.number_input("C (Regularization)", 0.01, 100.0, 1.0, 0.1)
            kernel = st.sidebar.selectbox("Kernel type", ["rbf", "linear"])
            model = SVR(C=c_val, kernel=kernel)
            
    else: # Clustering
        algorithm = st.sidebar.selectbox("Select Algorithm", ["K-Means", "DBSCAN"])
        if algorithm == "K-Means":
            n_clusters = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4, 1)
            model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        else:
            eps = st.sidebar.slider("Epsilon (Radius)", 0.1, 10.0, 1.0, 0.1)
            min_samples = st.sidebar.slider("Min Samples in Neighborhood", 2, 20, 5, 1)
            model = DBSCAN(eps=eps, min_samples=min_samples)

    if task_type in ("Classification", "Regression") and len(features) == 0:
        st.error("⚠️ Please select at least 1 feature variable (X) in the sidebar to continue.")
        st.stop()
    elif task_type == "Clustering" and len(features) < 2:
        st.error("⚠️ Please select at least 2 features in the sidebar for clustering visualization to continue.")
        st.stop()

    # Prepare input matrices
    X = df[features].copy()
    if X.isnull().sum().sum() > 0:
        st.warning("⚠️ Input features contain missing values. Filling missing values with column mean.")
        X = X.fillna(X.mean())

    # Scale if selected
    scaler = None
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values

    # Trigger Model Training
    train_clicked = st.sidebar.button("🚀 Train & Evaluate Model", type="primary")

    if train_clicked:
        st.session_state.task_trained = task_type
        st.session_state.features_used = features
        st.session_state.scaler = scaler

        if task_type == "Classification":
            y = df[target_col].copy()
            if pd.api.types.is_numeric_dtype(y):
                y_encoded = np.asarray(y.values)
                class_names = [str(x) for x in np.unique(y_encoded)]
            else:
                codes, uniques = pd.factorize(y)
                y_encoded = np.asarray(codes)
                class_names = [str(u) for u in uniques]

            unique_vals = np.unique(y_encoded)
            st.session_state.class_names = class_names
            st.session_state.unique_vals = unique_vals

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_encoded, test_size=test_size, random_state=42
                )
            
            model.fit(X_train, y_train)
            st.session_state.trained_model = model
            
        elif task_type == "Regression":
            y = df[target_col].copy()
            try:
                y_values = np.asarray(y.values, dtype=float)
            except (ValueError, TypeError):
                st.error(f"⚠️ Target column '{target_col}' isn't numeric.")
                st.stop()
            
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_values, test_size=test_size, random_state=42)
            model.fit(X_train, y_train)
            st.session_state.trained_model = model

        else: # Clustering
            labels = model.fit_predict(X_scaled)
            st.session_state.trained_model = model

        # ---- RENDER PERFORMANCE DASHBOARD ON TRAIN CLICK ----
        st.markdown("---")
        st.subheader(f"⚡ Model Evaluation Dashboard: {algorithm}")
        
        if task_type == "Classification":
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.4f}")
            m2.metric("Weighted Precision", f"{prec:.4f}")
            m3.metric("Weighted Recall", f"{rec:.4f}")
            m4.metric("Weighted F1-Score", f"{f1:.4f}")
            
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.write("📊 **Confusion Matrix**")
                cm = confusion_matrix(y_test, y_pred, labels=unique_vals)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
                
            with v_col2:
                st.write("📈 **ROC Curves**")
                fig, ax = plt.subplots(figsize=(5, 4))
                if y_prob is not None and len(class_names) >= 2:
                    if len(class_names) == 2:
                        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1], pos_label=unique_vals[1])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, label=f'{class_names[1]} (AUC = {roc_auc:.2f})')
                    else:
                        for i, name in enumerate(class_names):
                            y_test_bin = (y_test == unique_vals[i]).astype(int)
                            fpr, tpr, _ = roc_curve(y_test_bin, y_prob[:, i])
                            roc_auc = auc(fpr, tpr)
                            ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.legend(loc="lower right")
                    st.pyplot(fig)
                else:
                    st.info("ROC Curve requires multi-class probability scores.")
            
            if hasattr(model, "feature_importances_"):
                st.write("🎯 **Feature Importance**")
                feat_imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.barplot(x=feat_imp.values, y=feat_imp.index, hue=feat_imp.index, palette="viridis", legend=False, ax=ax)
                st.pyplot(fig)

        elif task_type == "Regression":
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("R² Score", f"{r2:.4f}")
            m2.metric("MSE", f"{mse:.4f}")
            m3.metric("RMSE", f"{rmse:.4f}")
            m4.metric("MAE", f"{mae:.4f}")
            
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.write("🎯 **Actual vs. Predicted Plot**")
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.scatter(y_test, y_pred, alpha=0.6, color='purple')
                ideal_line = [min(y_test), max(y_test)]
                ax.plot(ideal_line, ideal_line, 'r--', lw=2, label="y=x")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)
                
            with v_col2:
                st.write("📉 **Residuals Distribution Plot**")
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.histplot(residuals, kde=True, color='teal', ax=ax)
                ax.axvline(0, color='red', linestyle='--')
                st.pyplot(fig)

        else: # Clustering
            n_unique_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            m1, m2 = st.columns(2)
            m1.metric("Identified Clusters", n_unique_clusters)
            
            if n_unique_clusters > 1:
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > 5:
                    sil = silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask])
                    m2.metric("Silhouette Score", f"{sil:.4f}")

            st.write("📊 **Cluster Visualization (2D Projection)**")
            if len(features) > 2:
                pca = PCA(n_components=2)
                X_proj = pca.fit_transform(X_scaled)
                plot_x, plot_y = X_proj[:, 0], X_proj[:, 1]
                xlabel, ylabel = "PCA Component 1", "PCA Component 2"
            else:
                plot_x, plot_y = X[features[0]].values, X[features[1]].values
                xlabel, ylabel = features[0], features[1]
                
            fig, ax = plt.subplots(figsize=(8, 5))
            scatter = ax.scatter(plot_x, plot_y, c=labels, cmap='rainbow', alpha=0.8, edgecolors='k', s=50)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            plt.colorbar(scatter)
            st.pyplot(fig)

    # ---- NEW FEATURE: LIVE USER INPUT & PREDICTION OUTPUT ----
    if st.session_state.trained_model is not None and st.session_state.task_trained == task_type:
        st.markdown("---")
        st.subheader("🔮 Live Inference: Predict on New Data")
        st.write("Adjust the features below to view instant model predictions.")

        # Dynamically create input widgets for selected features based on dataset ranges
        user_inputs = {}
        cols = st.columns(min(len(st.session_state.features_used), 4))
        
        for idx, feat in enumerate(st.session_state.features_used):
            col = cols[idx % 4]
            min_val = float(df[feat].min())
            max_val = float(df[feat].max())
            mean_val = float(df[feat].mean())
            step_val = float((max_val - min_val) / 100) if max_val != min_val else 0.1
            
            user_inputs[feat] = col.number_input(
                f"{feat}",
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=step_val
            )

        # Convert user inputs to DataFrame
        input_df = pd.DataFrame([user_inputs])

        # Apply Scaling transformations matching the training configuration
        if st.session_state.scaler is not None:
            input_scaled = st.session_state.scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        # Make Prediction
        live_model = st.session_state.trained_model
        
        if task_type in ("Classification", "Regression"):
            prediction = live_model.predict(input_scaled)
            
            out_box = st.container(border=True)
            if task_type == "Classification":
                # Convert numeric index class back to actual name if factorized
                if st.session_state.class_names is not None:
                    pred_class = st.session_state.class_names[int(prediction[0])]
                else:
                    pred_class = str(prediction[0])
                
                out_box.markdown(f"### 🎯 Predicted Class: **`{pred_class}`**")
                
                # Show Probabilities if available
                if hasattr(live_model, "predict_proba"):
                    probs = live_model.predict_proba(input_scaled)[0]
                    prob_df = pd.DataFrame({
                        "Class": st.session_state.class_names,
                        "Probability (%)": [round(p * 100, 2) for p in probs]
                    })
                    out_box.dataframe(prob_df, hide_index=True, use_container_width=True)
            
            else: # Regression
                out_box.markdown(f"### 📈 Predicted Continuous Value: **`{prediction[0]:.4f}`**")
        
        else: # Clustering
            # Certain clustering models like DBSCAN do not predict on individual new points out-of-the-box
            if hasattr(live_model, "predict"):
                cluster_pred = live_model.predict(input_scaled)
                st.markdown(f"### 🏷️ Assigned Cluster ID: **`{cluster_pred[0]}`**")
            else:
                st.info("ℹ️ The selected clustering algorithm (e.g., DBSCAN) identifies core structures from historical structures and doesn't handle individual dynamic points natively.")

# Instructions on running this locally
st.markdown("---")
st.subheader("💻 Run this Streamlit Dashboard locally")
st.markdown("""
To run this application on your local machine, complete the following steps:

1. **Install required libraries:**
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn""")