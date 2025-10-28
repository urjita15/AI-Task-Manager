import streamlit as st
import pandas as pd
import joblib
import os
import json # Import json for correct metrics loading
from datetime import datetime, date
from scipy.sparse import hstack, vstack, csr_matrix # vstack added for correction
import numpy as np

ARTIFACT_DIR = "artifacts"

@st.cache_data
def load_artifacts():
    """Load models, vectorizer, encoders, metrics, and data."""
    try:
        tfidf = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
        nb = joblib.load(os.path.join(ARTIFACT_DIR, "nb_model.joblib"))
        svm = joblib.load(os.path.join(ARTIFACT_DIR, "svm_model.joblib"))
        rf = joblib.load(os.path.join(ARTIFACT_DIR, "rf_model.joblib"))
        cat_le = joblib.load(os.path.join(ARTIFACT_DIR, "cat_label_encoder.joblib"))
        pri_le = joblib.load(os.path.join(ARTIFACT_DIR, "pri_label_encoder.joblib"))
        
        # FIX: Load metrics as a Python dictionary using 'json', NOT 'pd.read_json', 
        # [cite_start]as it was saved using json.dump[cite: 1122, 1165].
        with open(os.path.join(ARTIFACT_DIR, "metrics_report.json"), "r") as f:
            metrics_dict = json.load(f)

        df = pd.read_csv(os.path.join(ARTIFACT_DIR, "tasks_synthetic.csv"))
        return tfidf, nb, svm, rf, cat_le, pri_le, metrics_dict, df
    except FileNotFoundError as e:
        st.error(f"Error loading artifact: {e}. Please ensure all model files are in the '{ARTIFACT_DIR}' folder.")
        st.stop()

# Load all required components
tfidf, nb, svm, rf, cat_le, pri_le, metrics, df = load_artifacts()

# FIX: Removed the unnecessary setup text that appeared before the content (Image 1 fix)
st.set_page_config(page_title="AI Task Manager", layout="wide")
st.title("AI-Powered Task Management System")

PAGES = ["Add New Task", "View All Tasks", "Workload Analyzer", "Prioritize & Manage", "Insights / Models"]
page = st.sidebar.radio("Pages", PAGES)

# Helper functions
def compute_features(desc, deadline_str, assigned_user, df_local):
    """Calculate raw numeric features for a task."""
    try:
        deadline_dt = datetime.fromisoformat(deadline_str).date()
    except:
        deadline_dt = date.fromisoformat(deadline_str)
        
    today = datetime.now().date()
    days_left = (deadline_dt - today).days
    task_length = len(str(desc).split())
    
    # Calculate current workload based on open/in progress tasks
    active = df_local[df_local['status'].isin(['Open', 'In Progress'])]
    workload_map = active['assigned_user'].value_counts().to_dict()
    user_workload = int(workload_map.get(assigned_user, 0))
    
    return days_left, task_length, user_workload

def vectorize(desc, days_left, task_length, user_workload):
    """Convert a task's description and numeric features into a single sparse matrix row."""
    X_text = tfidf.transform([desc])
    X_num = csr_matrix(np.array([[days_left, task_length, user_workload]], dtype=float))
    return hstack([X_text, X_num])


# Page: Add New Task
if page == "Add New Task":
    st.header("Add New Task")
    with st.form("add_task_form"):
        title = st.text_input("Task title") # Currently unused, assuming 'description' holds the main text
        description = st.text_area("Task description", height=120)
        deadline = st.date_input("Deadline", value=datetime.now().date())
        assigned_user = st.selectbox("Assigned user", sorted(df['assigned_user'].unique()))
        submit = st.form_submit_button("Submit")

        if submit:
            new_id = f"T{len(df)+1:05d}"
            
            # Compute raw features
            days_left, task_length, user_workload = compute_features(
                description, deadline.isoformat(), assigned_user, df
            )
            
            # FIX: Apply non-negative clipping to match training data
            days_left_nonneg = max(0, days_left)
            task_length_nonneg = max(0, task_length)
            user_workload_nonneg = max(0, user_workload)
            
            # Vectorize using clipped features
            X = vectorize(description, days_left_nonneg, task_length_nonneg, user_workload_nonneg)

            # Predict Category and Priority
            cat_pred = svm.predict(X)[0]
            pri_pred = rf.predict(X)[0]
            
            new_row = {
                "task_id": new_id,
                "description": description,
                "deadline": deadline.isoformat(),
                "assigned_user": assigned_user,
                "category": cat_le.inverse_transform([cat_pred])[0],
                "priority": pri_le.inverse_transform([pri_pred])[0],
                "status": "Open"
            }
            
            # Append new task to DataFrame and save
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(os.path.join(ARTIFACT_DIR, "tasks_synthetic.csv"), index=False)
            
            st.success("Task added successfully. Predicted Category and Priority are applied.")
            st.write(pd.DataFrame([new_row]).T)

# Page: View All Tasks
elif page == "View All Tasks":
    st.header("View All Tasks")
    
    cols = st.columns(4)
    user_f = cols[0].selectbox("Filter by User", options=["All"] + sorted(df['assigned_user'].unique()))
    pri_f = cols[1].selectbox("Filter by Priority", options=["All"] + sorted(df['priority'].unique()))
    status_f = cols[2].selectbox("Filter by Status", options=["All"] + sorted(df['status'].unique()))
    search = cols[3].text_input("Search description")
    
    df_view = df.copy()
    
    if user_f != "All": df_view = df_view[df_view['assigned_user']==user_f]
    if pri_f != "All": df_view = df_view[df_view['priority']==pri_f]
    if status_f != "All": df_view = df_view[df_view['status']==status_f]
    if search: df_view = df_view[df_view['description'].str.contains(search, case=False, na=False)]
    
    today = datetime.now().date()
    df_view['deadline'] = pd.to_datetime(df_view['deadline']).dt.date
    
    # Apply styling for overdue tasks
    def color_overdue(row):
        return ['background-color: #ffcccc' if row['deadline'] < today and row['status'] != 'Closed' else '' for _ in row]

    st.dataframe(df_view.style.apply(color_overdue, axis=1), use_container_width=True)


# Page: Workload Analyzer
elif page == "Workload Analyzer":
    st.header("Workload Analyzer")
    
    workload = df[df['status'].isin(["Open", "In Progress"])]['assigned_user'].value_counts()
    
    if not workload.empty:
        st.subheader("Current Workload (Active Tasks)")
        st.bar_chart(workload)
    else:
        st.info("No active tasks found to analyze workload.")

    st.subheader("Auto-assign new task")
    # FIX: Removed confusing st.selectbox and fixed the f-string syntax in the success message
    if st.button("Auto-assign next high-priority"):
        counts = workload.to_dict()
        for u in sorted(df['assigned_user'].unique()): counts.setdefault(u, 0)
        
        # Find the least busy user
        least_busy = min(counts.items(), key=lambda x: x[1])[0]
        # FIX: Corrected f-string syntax (was missing curly braces around variables)
        st.success(f"Recommendation: Assign to **{least_busy}** (active tasks: {counts[least_busy]})")


# Page: Prioritize & Manage Tasks
elif page == "Prioritize & Manage":
    st.header("Prioritize & Manage Tasks")
    
    if df.empty:
        st.info("No tasks available to prioritize or manage.")
    else:
        task_sel = st.selectbox("Select Task ID", options=df['task_id'].tolist())
        task_row = df[df['task_id']==task_sel].iloc[0]
        
        # FIX: Replaced task_row_to_frame().T with correct pandas syntax (Image 2 fix)
        st.write(pd.DataFrame([task_row]).T)
        
        current_priority = task_row['priority']
        current_status = task_row['status']
        
        new_priority = st.selectbox(
            "New priority", 
            ["High", "Medium", "Low"], 
            index=["High", "Medium", "Low"].index(current_priority)
        )
        new_status = st.selectbox(
            "New status", 
            ["Open", "In Progress", "Closed"], 
            index=["Open", "In Progress", "Closed"].index(current_status)
        )
        
        if st.button("Update task"):
            df.loc[df['task_id'] == task_sel, 'priority'] = new_priority
            df.loc[df['task_id'] == task_sel, 'status'] = new_status
            df.to_csv(os.path.join(ARTIFACT_DIR, "tasks_synthetic.csv"), index=False)
            st.success("Task updated.")

        if st.button("Recalculate priorities"):
            updated = 0
            
            # Recalculate priorities for all open/in progress tasks
            for idx, row in df.iterrows():
                if row['status'] not in ['Open', 'In Progress']:
                    continue
                    
                # Compute raw features using the centralized function
                days_left, task_length, user_workload = compute_features(
                    row['description'], row['deadline'], row['assigned_user'], df
                )

                # FIX: Apply non-negative clipping to match training data
                days_left_nonneg = max(0, days_left)
                task_length_nonneg = max(0, task_length)
                user_workload_nonneg = max(0, user_workload)
                
                # Vectorize using clipped features
                X = vectorize(row['description'], days_left_nonneg, task_length_nonneg, user_workload_nonneg)
                
                # Predict new priority
                new_pri_encoded = rf.predict(X)[0]
                new_pri = pri_le.inverse_transform([new_pri_encoded])[0]
                
                if df.at[idx, 'priority'] != new_pri:
                    df.at[idx, 'priority'] = new_pri
                    updated += 1

            df.to_csv(os.path.join(ARTIFACT_DIR, "tasks_synthetic.csv"), index=False)
            st.success(f"Updated {updated} tasks with new predicted priorities.")


# Page: Insights / Models
elif page == "Insights / Models":
    st.header("Insights & Model Performance")
    
    st.subheader("Model metrics")
    st.json(metrics) # metrics is now correctly loaded as a dict
    
    st.subheader("Category distribution")
    st.write(df['category'].value_counts())
    
    st.subheader("Priority distribution")
    st.write(df['priority'].value_counts())
    
    st.subheader("Sample category predictions (SVM)")
    
    # Take a random sample from the current DataFrame
    sample = df.sample(min(50, len(df)), random_state=42)

    # Prepare features for the sample: Create a list of single-row sparse matrices
    sample_features = []
    for idx, r in sample.iterrows():
        # Compute raw features
        days_left, task_length, user_workload = compute_features(
            r['description'], r['deadline'], r['assigned_user'], df
        )
        
        # FIX: Apply non-negative clipping to match training data
        days_left_nonneg = max(0, days_left)
        task_length_nonneg = max(0, task_length)
        user_workload_nonneg = max(0, user_workload)
        
        # Vectorize single task
        X_single = vectorize(r['description'], days_left_nonneg, task_length_nonneg, user_workload_nonneg)
        sample_features.append(X_single)
    
    # FIX: Use vstack to combine single-row feature vectors into a single test set matrix
    X_sample = vstack(sample_features) 

    # Get predictions
    preds = [cat_le.inverse_transform([int(p)])[0] for p in svm.predict(X_sample)]
    
    # Display results
    sample_display = sample.copy()
    sample_display["svm_pred_category"] = preds
    st.dataframe(sample_display[['task_id', 'description', 'category', 'svm_pred_category']], use_container_width=True)
