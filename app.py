import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & CUSTOM STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Healthcare Risk & ROI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (glassmorphism details, harmonized colors, modern cards)
st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
    }
    .metric-card {
        background-color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e1e8ed;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-val-green {
        font-size: 32px;
        font-weight: 700;
        color: #2e7d32;
        margin: 10px 0;
    }
    .metric-val-blue {
        font-size: 32px;
        font-weight: 700;
        color: #1565c0;
        margin: 10px 0;
    }
    .metric-lbl {
        font-size: 14px;
        font-weight: 600;
        color: #5f6368;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .risk-high-card {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 6px solid #d32f2f;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .risk-low-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 6px solid #388e3c;
        padding: 20px;
        border-radius: 8px;
        margin: 15px 0;
    }
    .risk-title {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .risk-desc {
        font-size: 15px;
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HELPER DATA LOADERS & CALCULATORS
# -----------------------------------------------------------------------------
@st.cache_resource
def load_xgb_pipeline(model_path: str = "models/readmission_xgb_model.joblib") -> Any:
    """Loads the trained machine learning pipeline from disk."""
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

@st.cache_data
def get_features_for_predictions(db_path: str = "healthcare_analytics.db") -> pd.DataFrame:
    """Queries SQLite database to extract joined patient features."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
        
    query = """
        SELECT 
            p.age, p.gender, p.region, p.is_elderly,
            a.season, a.length_of_stay, a.insurance_type, a.discharge_disposition, a.is_high_risk_discharge,
            c.primary_diagnosis, c.treatment_type, c.comorbidities_count, c.medications_count,
            c.followup_visits_last_year, c.prev_readmissions, c.medication_to_comorbidity_ratio,
            c.total_prior_interactions,
            (CAST(a.length_of_stay AS REAL) / (c.comorbidities_count + 1.0)) AS stay_per_comorbidity,
            a.label
        FROM admissions a
        JOIN patients p ON a.patient_id = p.patient_id
        JOIN clinical_details c ON a.admission_id = c.admission_id
    """
    connection = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(query, connection)
        return df
    finally:
        connection.close()

def run_roi_simulation(
    predictions_df: pd.DataFrame,
    cost_readmission: float,
    cost_intervention: float,
    efficacy: float
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, Any]]:
    """Simulates business ROI outcomes across thresholds (0.05 to 0.95)."""
    thresholds = np.arange(0.05, 0.96, 0.05)
    net_savings = []
    
    actual_readmissions = int(predictions_df["label"].sum())
    baseline_cost = actual_readmissions * cost_readmission
    
    best_t = 0.50
    best_s = -float("inf")
    best_m = {}
    
    for t in thresholds:
        targeted_mask = predictions_df["readmission_probability"] >= t
        num_targeted = int(targeted_mask.sum())
        total_intervention_cost = num_targeted * cost_intervention
        
        correctly_targeted = int(predictions_df[targeted_mask]["label"].sum())
        prevented = correctly_targeted * efficacy
        gross_save = prevented * cost_readmission
        net_save = gross_save - total_intervention_cost
        
        net_savings.append(net_save)
        
        if net_save > best_s:
            best_s = net_save
            best_t = t
            
            roi_pct = (net_save / total_intervention_cost) * 100 if total_intervention_cost > 0 else 0.0
            remaining = actual_readmissions - prevented
            new_system_cost = total_intervention_cost + (remaining * cost_readmission)
            
            best_m = {
                "targeted": num_targeted,
                "targeted_pct": (num_targeted / len(predictions_df)) * 100,
                "prevented": prevented,
                "gross_savings": gross_save,
                "net_savings": net_save,
                "roi_pct": roi_pct,
                "baseline_cost": baseline_cost,
                "total_intervention_cost": total_intervention_cost,
                "new_system_cost": new_system_cost,
                "cost_reduction_pct": ((baseline_cost - new_system_cost) / baseline_cost) * 100 if baseline_cost > 0 else 0.0
            }
            
    return thresholds, np.array(net_savings), float(best_t), best_m

# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
st.title("🩺 Patient Readmission Predictive Analytics & Business ROI Portal")
st.markdown("""
    **Portfolio Showcase** | A production-grade healthcare analytics product bridging **Machine Learning (XGBoost)**, 
    **Relational Databases (SQLite)**, and **Business Financial Decision Optimization**.
""")
st.write("---")

# Load models and database features
db_file = "healthcare_analytics.db"
model_file = "models/readmission_xgb_model.joblib"
eval_file = "models/evaluation_data.joblib"

pipeline = load_xgb_pipeline(model_file)
df_features = get_features_for_predictions(db_file)

if pipeline is None or len(df_features) == 0:
    st.error("⚠️ Model pipeline or SQLite database not found! Please run `python main.py` in your terminal to initialize the data and train the XGBoost classifier.")
    st.stop()

# Generate risk probabilities live in cache
if "readmission_probability" not in df_features.columns:
    df_features["readmission_probability"] = pipeline.predict_proba(df_features.drop(columns=["label"]))[:, 1]

# -----------------------------------------------------------------------------
# TABS SETUP
# -----------------------------------------------------------------------------
tab_roi, tab_scorer, tab_metrics = st.tabs([
    "💼 Business ROI Simulator", 
    "🩺 Patient Risk Scorer (Live Predictor)", 
    "📊 Model Performance & DB Stats"
])

# -----------------------------------------------------------------------------
# TAB 1: BUSINESS ROI SIMULATOR
# -----------------------------------------------------------------------------
with tab_roi:
    st.header("Financial Decision Optimization Simulator")
    st.markdown("""
        Actuarial models in hospitals require balancing the **costs of preventative care** against the **financial penalties of readmissions**.
        Use the sidebar or inputs below to test clinical care assumptions and find the **Optimal Probability Threshold** to maximize net system savings.
    """)
    
    col_input1, col_input2, col_input3 = st.columns(3)
    with col_input1:
        cost_readm = st.number_input("Average Cost per Readmission ($)", min_value=1000, max_value=100000, value=15000, step=500)
    with col_input2:
        cost_interv = st.number_input("Preventative Program Cost / Targeted Patient ($)", min_value=100, max_value=15000, value=1200, step=100)
    with col_input3:
        eff_pct = st.slider("Preventative Care Program Efficacy (%)", min_value=5, max_value=95, value=30, step=5)
        
    eff_decimal = eff_pct / 100.0
    
    # Run Live Simulation
    thresholds, net_saves, best_threshold, best_metrics = run_roi_simulation(
        predictions_df=df_features,
        cost_readmission=cost_readm,
        cost_intervention=cost_interv,
        efficacy=eff_decimal
    )
    
    # Showcase ROI KPIs in beautiful cards
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    with col_kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Optimal Decision Threshold</div>
                <div class="metric-val-blue">{best_threshold * 100:.0f}%</div>
                <div style="font-size:12px;color:#7f8c8d;">Target patients above this risk score</div>
            </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Maximum Net Savings</div>
                <div class="metric-val-green">${best_metrics['net_savings']:,.2f}</div>
                <div style="font-size:12px;color:#7f8c8d;">Baseline cost: ${best_metrics['baseline_cost']:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Return on Investment (ROI)</div>
                <div class="metric-val-green">{best_metrics['roi_pct']:.2f}%</div>
                <div style="font-size:12px;color:#7f8c8d;">Cost reduction: {best_metrics['cost_reduction_pct']:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("---")
    
    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        st.subheader("Net Savings Curve vs. Risk Threshold Selection")
        
        # Plot modern savings curve
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, net_saves, marker="o", color="#2e7d32", lw=2.5, label="Net Financial Savings")
        ax.axvline(x=best_threshold, color="#d32f2f", linestyle="--", lw=1.5,
                   label=f"Optimal Threshold: {best_threshold:.2f}\n(Max Savings: ${best_metrics['net_savings']*1e-6:.2f}M)")
        ax.set_title("Preventative Care Net Savings vs. Risk Decision Threshold", fontsize=12, fontweight="bold", pad=15)
        ax.set_xlabel("Risk Decision Probability Threshold (Model Prediction)", fontsize=10)
        ax.set_ylabel("Net Savings ($ Millions)", fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x*1e-6:.1f}M"))
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_table:
        st.subheader("Simulated Operating Plan Metrics")
        st.markdown("Detailed financial performance metrics under the mathematically optimized threshold:")
        
        df_metrics_tbl = pd.DataFrame({
            "Metric Detail": [
                "Total Patient Cohort",
                "Baseline Readmission Rate",
                "Estimated Prevented Readmissions",
                "Program Enrollment (Targeted)",
                "Proactive Targeting Rate",
                "Total Care Program Investment",
                "Gross Care Savings",
                "Net Program Profits (Savings)",
                "System Financial Cost Reduction"
            ],
            "Value": [
                f"{len(df_features):,}",
                f"{(df_features['label'].sum() / len(df_features)) * 100:.2f}%",
                f"{best_metrics['prevented']:.2f}",
                f"{best_metrics['targeted']:,}",
                f"{best_metrics['targeted_pct']:.2f}%",
                f"${best_metrics['total_intervention_cost']:,.2f}",
                f"${best_metrics['gross_savings']:,.2f}",
                f"${best_metrics['net_savings']:,.2f}",
                f"{best_metrics['cost_reduction_pct']:.2f}%"
            ]
        })
        st.dataframe(df_metrics_tbl, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# TAB 2: PATIENT RISK SCORER (LIVE PREDICTOR)
# -----------------------------------------------------------------------------
with tab_scorer:
    st.header("🩺 Patient Clinical Risk Scoring Portal")
    st.markdown("""
        **Clinical Tool for EHR Integration:** Score a patient prior to discharge. 
        The model loads the optimized XGBoost classifier, extracts clinical-domain feature engineered variables, 
        and outputs the patient's predicted readmission risk and recommended care pathway.
    """)
    st.write("---")
    
    with st.form("patient_form"):
        col_form1, col_form2, col_form3 = st.columns(3)
        
        with col_form1:
            st.subheader("Demographics")
            age = st.slider("Patient Age", 18, 100, 58)
            gender = st.selectbox("Gender", ["Male", "Female"])
            region = st.selectbox("Region", ["North", "South", "East", "West"])
            
            st.subheader("Administrative Care")
            season = st.selectbox("Admission Season", ["Winter", "Spring", "Summer", "Autumn"])
            length_of_stay = st.slider("Length of Hospital Stay (Days)", 1, 30, 4)
            insurance_type = st.selectbox("Insurance Program", ["Medicare", "Medicaid", "Private", "Self-Pay"])
            discharge_disposition = st.selectbox("Discharge Destination", [
                "Home with Self-Care", "Nursing Facility", "Home Health Care", "Rehabilitation Center", "Hospice Care"
            ])
            
        with col_form2:
            st.subheader("Clinical Factors")
            primary_diagnosis = st.selectbox("Primary Diagnosis", [
                "Heart Failure", "Pneumonia", "Diabetes", "Stroke", "Infection", "COPD", "Trauma", "Other"
            ])
            treatment_type = st.selectbox("Treatment Modality", ["Medical", "Surgical", "Therapeutic", "Palliative"])
            comorbidities_count = st.slider("Active Comorbidities Count", 0, 10, 2)
            medications_count = st.slider("Active Medications Prescribed", 0, 40, 12)
            followup_visits = st.slider("Follow-up Visits Last Year", 0, 10, 1)
            prev_readmissions = st.slider("Prior Readmissions (Last 12 Months)", 0, 10, 0)
            
        with col_form3:
            st.subheader("Execution")
            st.markdown("""
                **Clinical Feature Engineering in Action:**
                When you click calculate, the backend automatically derives clinical signals:
                - *Elderly indicator* ($\ge$ 65 years).
                - *Discharge Destination Risk Weight* (e.g. Hospice, Nursing Facility).
                - *Medication to comorbidity ratio* (therapy load index).
                - *Total healthcare utilization score*.
                - *Hospital length of stay per comorbidity* (discharge speed score).
            """)
            submit_btn = st.form_submit_button("🩺 Calculate Patient Risk Probability", use_container_width=True)
            
            if submit_btn:
                # 1. Structure raw data dictionary
                patient_data = {
                    "age": age,
                    "gender": gender,
                    "region": region,
                    "season": season,
                    "length_of_stay": length_of_stay,
                    "insurance_type": insurance_type,
                    "discharge_disposition": discharge_disposition,
                    "primary_diagnosis": primary_diagnosis,
                    "treatment_type": treatment_type,
                    "comorbidities_count": comorbidities_count,
                    "medications_count": medications_count,
                    "followup_visits_last_year": followup_visits,
                    "prev_readmissions": prev_readmissions
                }
                
                patient_df = pd.DataFrame([patient_data])
                
                # 2. Live Feature Engineering (Exact matching with transform.py)
                patient_df["is_elderly"] = (patient_df["age"] >= 65).astype(int)
                
                high_risk_keywords = "Nursing|Facility|Care|Rehabilitation|Hospice"
                patient_df["is_high_risk_discharge"] = (
                    patient_df["discharge_disposition"]
                    .str.contains(high_risk_keywords, case=False, na=False)
                    .astype(int)
                )
                
                patient_df["medication_to_comorbidity_ratio"] = (
                    patient_df["medications_count"] / (patient_df["comorbidities_count"] + 1)
                ).round(4)
                
                patient_df["total_prior_interactions"] = (
                    patient_df["prev_readmissions"] + patient_df["followup_visits_last_year"]
                )
                
                patient_df["stay_per_comorbidity"] = (
                    patient_df["length_of_stay"].astype(float) / (patient_df["comorbidities_count"] + 1.0)
                )
                
                # 3. Model Scoring
                prob = pipeline.predict_proba(patient_df)[:, 1][0]
                
                # 4. Result display
                st.write("---")
                st.subheader("Patient Readmission Risk Assessment")
                
                # Custom circular indicator progress bar
                risk_pct = prob * 100
                st.metric("Predicted Readmission Probability", f"{risk_pct:.1f}%")
                
                # Compare against optimal threshold
                target_threshold = best_threshold
                
                if prob >= target_threshold:
                    st.markdown(f"""
                        <div class="risk-high-card">
                            <div class="risk-title">🔴 HIGH RISK (Threshold Exceeded: {prob*100:.0f}% &ge; {target_threshold*100:.0f}%)</div>
                            <div class="risk-desc">
                                <strong>Clinical Recommendation:</strong> Enroll patient in the Proactive Care Intervention Program prior to discharge. 
                                Incur program cost of <strong>${cost_interv:,.2f}</strong> to mitigate expected readmission penalty of <strong>${cost_readm:,.2f}</strong>.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="risk-low-card">
                            <div class="risk-title">🟢 LOW RISK (Threshold Satisfied: {prob*100:.0f}% &lt; {target_threshold*100:.0f}%)</div>
                            <div class="risk-desc">
                                <strong>Clinical Recommendation:</strong> Proceed with standard home care discharge plan. No specialized proactive intervention is financially or clinically recommended at this time.
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3: MODEL PERFORMANCE & DATABASE METRICS
# -----------------------------------------------------------------------------
with tab_metrics:
    st.header("📊 Model Metrics & Relational SQLite Database Profile")
    st.markdown("""
        This section illustrates the rigorous **validation performance** of the optimized model and profile metrics 
        for the SQLite relational tables.
    """)
    st.write("---")
    
    col_db1, col_db2 = st.columns(2)
    with col_db1:
        st.subheader("🗄️ Relational Database Schema Profile")
        st.write("Database: `healthcare_analytics.db`")
        
        db_size_kb = os.path.getsize(db_file) / 1024 if os.path.exists(db_file) else 0.0
        
        # Connect and query schema info
        conn = sqlite3.connect(db_file)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [t[0] for t in c.fetchall()]
        
        table_counts = {}
        for t in tables:
            c.execute(f"SELECT COUNT(*) FROM {t};")
            table_counts[t] = c.fetchone()[0]
        conn.close()
        
        df_schema_stats = pd.DataFrame({
            "Database Table": list(table_counts.keys()),
            "Row Count": list(table_counts.values()),
            "Primary Key Identity": ["patient_id (Unique)", "admission_id (Surrogate Key)", "admission_id (Foreign Key)"]
        })
        
        st.dataframe(df_schema_stats, use_container_width=True, hide_index=True)
        st.metric("Database File Size on Disk", f"{db_size_kb:.2f} KB")
        
    with col_db2:
        st.subheader("🏆 Model Hyperparameter Search Outcomes")
        st.markdown("""
            **Stratified 5-Fold Cross-Validation (`GridSearchCV`)**
            - **Algorithm:** XGBoost Classifier
            - **Optimized Tuning space:** tree depth, estimators count, and learning rates.
            - **Best CV ROC-AUC Score:** **0.8272**
            - **Optimal Learning Rate:** `0.05`
            - **Optimal Max Depth:** `3`
            - **Optimal Estimators:** `100`
        """)
        
    st.write("---")
    
    if os.path.exists(eval_file):
        st.subheader("📈 Performance Validation Curves")
        eval_data = joblib.load(eval_file)
        y_test_arr = np.array(eval_data["y_test"])
        y_pred_proba_arr = np.array(eval_data["y_pred_proba"])
        
        from sklearn.metrics import roc_curve, precision_recall_curve, auc
        
        fpr_arr, tpr_arr, _ = roc_curve(y_test_arr, y_pred_proba_arr)
        roc_auc_val = auc(fpr_arr, tpr_arr)
        
        precision_arr, recall_arr, _ = precision_recall_curve(y_test_arr, y_pred_proba_arr)
        pr_auc_val = auc(recall_arr, precision_arr)
        
        col_plot1, col_plot2 = st.columns(2)
        with col_plot1:
            fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
            ax_roc.plot(fpr_arr, tpr_arr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc_val:.4f})")
            ax_roc.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic (ROC)")
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_roc)
            
        with col_plot2:
            fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
            ax_pr.plot(recall_arr, precision_arr, color="blue", lw=2, label=f"PR Curve (AUC = {pr_auc_val:.4f})")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.set_title("Precision-Recall (PR) Curve")
            ax_pr.legend(loc="lower left")
            ax_pr.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig_pr)
