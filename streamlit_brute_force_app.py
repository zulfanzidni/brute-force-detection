
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load model and example data
@st.cache_resource
def load_model():
    model = joblib.load("final_rf_brute_force_model.joblib")
    return model

model = load_model()

st.title("Brute Force Login Detection")

st.markdown("Upload log data (CSV) with columns: `timestamp`, `ip_address`, `username`, `status`.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by=['ip_address', 'timestamp'], inplace=True)
    df['failed_count_prev_1min'] = 0

    for ip, group in df.groupby('ip_address'):
        idxs = group.index
        times = group['timestamp']
        statuses = group['status']

        for i in range(len(group)):
            t_now = times.iloc[i]
            time_window = group[(times >= t_now - pd.Timedelta(minutes=1)) & (times < t_now)]
            failed_attempts = (time_window['status'] == 'failed').sum()
            df.loc[idxs[i], 'failed_count_prev_1min'] = failed_attempts

    # Prediction
    df['prediction'] = model.predict(df[['failed_count_prev_1min']])

    st.success("Detection complete. Rows flagged as brute force (1) are shown below.")
    st.dataframe(df[df['prediction'] == 1])
    st.download_button("Download Full Result", df.to_csv(index=False), "brute_force_detection_result.csv", "text/csv")
