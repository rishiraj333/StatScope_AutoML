import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error

# ---------------- DATETIME DETECTION ----------------
def detect_datetime_col(df_sample):
    col_types = {}
    date_patterns = [
        r'^\d{4}-\d{2}-\d{2}$', r'^\d{4}/\d{2}/\d{2}$',
        r'^\d{2}/\d{2}/\d{4}$', r'^\d{2}-\d{2}-\d{4}$',
        r'^\d{4}\.\d{2}\.\d{2}$', r'^\d{2}\.\d{2}\.\d{4}$',
        r'^\d{8}$', r'^\d{6}$'
    ]
    time_patterns = [r'^\d{2}:\d{2}$', r'^\d{2}:\d{2}:\d{2}$']
    datetime_patterns = [
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?$',
        r'^\d{4}/\d{2}/\d{2} \d{2}:\d{2}(:\d{2})?$',
        r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}(:\d{2})?$',
        r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$'
    ]
    for col in df_sample.columns:
        col_data = df_sample[col].dropna().astype(str).iloc[:10]
        for val in col_data:
            val = val.strip()
            matched_type = None
            for pat in datetime_patterns:
                if re.match(pat, val):
                    matched_type = 'datetime'
                    break
            if matched_type:
                col_types[col] = matched_type
                break
            for pat in date_patterns:
                if re.match(pat, val):
                    matched_type = 'date'
                    break
            if matched_type:
                col_types[col] = matched_type
                break
            for pat in time_patterns:
                if re.match(pat, val):
                    matched_type = 'time'
                    break
            if matched_type:
                col_types[col] = matched_type
                break
            if val.isdigit():
                ival = int(val)
                if len(val) == 8:
                    try:
                        pd.to_datetime(val, format='%Y%m%d')
                        col_types[col] = 'date'
                        break
                    except:
                        continue
                elif len(val) == 6:
                    try:
                        pd.to_datetime(val, format='%Y%m')
                        col_types[col] = 'date'
                        break
                    except:
                        continue
                elif ival > 1e9:
                    for unit in ['s', 'ms']:
                        try:
                            pd.to_datetime(ival, unit=unit)
                            col_types[col] = 'datetime'
                            break
                        except:
                            continue
                    if col in col_types:
                        break
            try:
                parsed = pd.to_datetime(val, errors='raise')
                if parsed.time() == pd.Timestamp.min.time():
                    col_types[col] = 'date'
                elif parsed.date() == pd.Timestamp.min.date():
                    col_types[col] = 'time'
                else:
                    col_types[col] = 'datetime'
                break
            except:
                continue
    return col_types

# ---------------- GRID SEARCH ----------------
def dynamic_grid_search_until_converge(X_train, y_train, X_val, y_val):
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    prev_r2 = -np.inf
    best_model = None
    best_params = None
    keep_searching = True
    max_iterations = 5
    iteration = 0

    def around(val, low=1, step=2):
        if val is None:
            return [None, 5, 10]
        if isinstance(val, int):
            return sorted(list(set([max(low, val - step), val, val + step])))
        return [val]

    while keep_searching and iteration < max_iterations:
        st.info(f"🔍 Iteration {iteration+1} grid search in progress...")
        grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='r2')
        grid.fit(X_train, y_train)
        this_model = grid.best_estimator_
        y_pred_val = this_model.predict(X_val)
        this_r2 = r2_score(y_val, y_pred_val)
        st.write(f"Iteration {iteration+1} R²: {this_r2:.4f}")
        if this_r2 > prev_r2:
            prev_r2 = this_r2
            best_model = this_model
            best_params = grid.best_params_
            param_grid = {
                'n_estimators': around(best_params['n_estimators'], 10, 20),
                'max_depth': around(best_params['max_depth'], 1, 2),
                'min_samples_split': around(best_params['min_samples_split'], 2, 1),
                'min_samples_leaf': around(best_params['min_samples_leaf'], 1, 1)
            }
            iteration += 1
        else:
            st.success(f"✅ Converged after {iteration} iterations.")
            keep_searching = False
    return best_model, best_params, prev_r2

# ---------------- STREAMLIT APP ----------------
st.title("📊 StatScope AutoML: Predictive Modelling for Dummies")

uploaded_file = st.file_uploader("Upload your CSV dataset:", type=['csv'])
if uploaded_file:
    # Load initially to detect datetime cols
    df_preview = pd.read_csv(uploaded_file, nrows=10)
    col_types = detect_datetime_col(df_preview)
    st.write("🕒 Detected date/time columns:", col_types)
    
    # Reset file pointer to the beginning
    uploaded_file.seek(0)  

    # Identify datetime candidates
    datetime_candidates = [col for col, typ in col_types.items() if typ in ['date', 'datetime']]
    
    # Choose first datetime column as index (or let user choose)
    datetime_col = None
    if datetime_candidates:
        datetime_col = datetime_candidates[0]
        st.info(f"📌 Using `{datetime_col}` as datetime index.")
        df = pd.read_csv(uploaded_file, parse_dates=[datetime_col], index_col=datetime_col)
    else:
        df = pd.read_csv(uploaded_file)
        st.warning("⚠️ No datetime column found to set as index.")

    # ✅ Add datetime features
    if datetime_col:
        # Ensure index is DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        # Now safe to extract datetime features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['weekday'] = df.index.weekday

    # Only add hour if time info is present
    if hasattr(df.index, 'hour'):
        df['hour'] = df.index.hour

    st.write("🕒 Added datetime-derived features: `year`, `month`, `day`, `weekday`, `hour` (if available).")
    # Display dataset info
    st.write("✅ Data loaded successfully! Here's a preview:")
    st.dataframe(df.head())

    # Nulls + Imputation
    st.write("🔎 Checking for nulls + duplicates...")
    nulls = df.columns[df.isnull().sum() > 0].tolist()
    if nulls:
        st.write(f"Missing values found in: {nulls} — imputing using KNN...")
        imputer = KNNImputer(n_neighbors=5)
        df[nulls] = imputer.fit_transform(df[nulls])
    else:
        st.write("✅ No missing values found.")

    dup_count = df.duplicated().sum()
    if dup_count:
        st.write(f"Found {dup_count} duplicate rows — removing...")
        df = df.drop_duplicates()
    else:
        st.write("✅ No duplicate rows found.")

    # Target selection
    target_col = st.selectbox("🎯 Select your target variable (what you want to predict):", df.columns)
    feature_cols = st.multiselect("🛠️ Select features for training the model:", 
                                  [col for col in df.columns if col != target_col], default=[col for col in df.columns if col != target_col])

    if not feature_cols:
        feature_cols = [col for col in df.columns if col != target_col]

    # Encode categoricals
    for col in feature_cols:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Scale features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df[feature_cols]), columns=feature_cols)
    y = df[target_col]

    # Time-based Train/Test Split: last 20% rows as test data
    split_idx = int(len(X) * 0.8)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Further split train into train2 / val for grid search (no shuffle, preserves order)
    val_split_idx = int(len(X_train) * 0.8)
    X_train2, X_val = X_train.iloc[:val_split_idx], X_train.iloc[val_split_idx:]
    y_train2, y_val = y_train.iloc[:val_split_idx], y_train.iloc[val_split_idx:]

    if st.button("🚀 Run AutoML"):
        best_model, best_params, best_r2 = dynamic_grid_search_until_converge(X_train2, y_train2, X_val, y_val)
        y_pred = best_model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        st.subheader("📈 Evaluation Metrics")
        st.write(f"**R² (Coefficient of Determination 0.0-1.0)**: {r2:.4f} — how well your model explains the variance in the data.")
        st.write(f"**MSE (Mean Squared Error)**: {mse:.4f} — average of the squares of the errors, lower is better.")

        # Plots
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(y_test.index, y_test, label='Actual Values', color='violet', linewidth=2)

        ax.plot(y_test.index, y_pred, label='Predicted Values', color='green', linestyle='dashed', linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel(target_col)  # dynamic label
        ax.set_title(f'Test Data: Actual vs Predicted {target_col}')
        ax.legend()

        plt.xticks(rotation=45)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        plt.tight_layout()
        st.pyplot(fig)

        # Download options
        model_bytes = pickle.dumps(best_model)
        st.download_button("💾 Download Trained Model", model_bytes, file_name="trained_model.pkl")

        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        csv = pred_df.to_csv(index=False).encode()
        st.download_button("💾 Download Predictions CSV", csv, file_name="predictions.csv")

        st.write("✅ AutoML process complete. Model and predictions are ready for use!")