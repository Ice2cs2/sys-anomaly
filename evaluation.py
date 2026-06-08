import streamlit as st

st.set_page_config(
    page_title="System Anomaly Detection",
    page_icon="🖥️",
    layout="wide"
)

st.markdown("""
<style>
    .block-container { padding: 2rem 3rem; }
    .metric-lbl { font-size: 11px; color: #94a3b8; font-weight: 700;
                  text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
    .metric-val { font-size: 26px; font-weight: 800; color: #0f172a; }
    .metric-sub { font-size: 12px; color: #64748b; margin-top: 2px; }
    .bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }
    .bar-lbl { font-size: 11px; color: #64748b; width: 80px; flex-shrink: 0; }
    .bar-bg  { flex: 1; height: 6px; background: #f1f5f9; border-radius: 4px; overflow: hidden; }
    .insight-item {
        display: flex; gap: 14px; align-items: flex-start;
        background: #f8fafc; border-radius: 12px;
        padding: 16px; margin-bottom: 10px;
    }
    .insight-icon {
        width: 40px; height: 40px; border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 20px; flex-shrink: 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Data ─────────────────────────────────────────────────────────────────────

MODELS = [
    {
        "id": 0,
        "name": "Isolation Forest",
        "subtitle": "Tree-based path-length scoring",
        "color": "#3B82F6",
        "bgLight": "#EFF6FF",
        "textColor": "#1E40AF",
        "precision": 84,
        "recall": 79,
        "f1": 81,
        "auc": 94,
        "goal": "ກວດຈັບ spike ແລະ ຜິດປົກກະຕິໄວທີ່ສຸດ",
        "insights": [
            {
                "icon": "⚡",
                "name": "Random Path Partitioning",
                "desc": "Anomalous system states (CPU spike, memory leak) are isolated with far fewer random splits "
                        "than normal states. Score = average path length across 300 trees.",
                "tag1": "Speed: Real-time",
                "tag2": "Effort: Low",
            },
            {
                "icon": "📊",
                "name": "Contamination = True Anomaly Rate",
                "desc": "contamination parameter tuned to actual anomaly rate from dataset (~2–5%). "
                        "Threshold auto-adjusts to flag only the most extreme deviations.",
                "tag1": "Precision: 84%",
                "tag2": "Recall: 79%",
            },
            {
                "icon": "🔁",
                "name": "No Label Required",
                "desc": "Trained entirely on unlabeled system telemetry. Labels were used ONLY for post-hoc "
                        "evaluation — the model never saw them during training.",
                "tag1": "Unsupervised: ✓",
                "tag2": "Scalable: ✓",
            },
        ]
    },
    {
        "id": 1,
        "name": "DBSCAN",
        "subtitle": "Density-based spatial clustering",
        "color": "#F59E0B",
        "bgLight": "#FFFBEB",
        "textColor": "#92400E",
        "precision": 76,
        "recall": 71,
        "f1": 73,
        "auc": 0,
        "goal": "ຊອກຫາ cluster ຜິດປົກກະຕິທີ່ຢູ່ໃນຂອບເຂດໜ້ອຍ",
        "insights": [
            {
                "icon": "🗺️",
                "name": "Noise Points = Anomalies",
                "desc": "Points in sparse regions that don't belong to any dense cluster are labeled -1 (noise). "
                        "These correspond to system states that never occur under normal load.",
                "tag1": "No score output",
                "tag2": "Hard labels only",
            },
            {
                "icon": "🔍",
                "name": "eps Tuned via k-Distance",
                "desc": "Optimal eps selected at the 'elbow' of the k-nearest-neighbor distance graph "
                        "(k = 2 × n_features). Prevents over-splitting normal dense regions.",
                "tag1": "Sensitivity: Medium",
                "tag2": "Effort: Medium",
            },
            {
                "icon": "⚠️",
                "name": "Limitation: High Dimensions",
                "desc": "DBSCAN degrades in high-dimensional feature spaces (curse of dimensionality). "
                        "Applied after PCA feature reduction to mitigate this.",
                "tag1": "F1: 73%",
                "tag2": "AUC-ROC: N/A",
            },
        ]
    },
    {
        "id": 2,
        "name": "Autoencoder",
        "subtitle": "Reconstruction error threshold",
        "color": "#8B5CF6",
        "bgLight": "#F5F3FF",
        "textColor": "#5B21B6",
        "precision": 88,
        "recall": 83,
        "f1": 85,
        "auc": 97,
        "goal": "ຈັບ pattern ລະອຽດໃນ system metric ທີ່ Model ອື່ນພາດ",
        "insights": [
            {
                "icon": "🧠",
                "name": "Trained on Normal Behavior Only",
                "desc": "Architecture: 32→16→8→16→32 with BatchNorm. Trained exclusively on normal system "
                        "snapshots. Anomalous states reconstruct poorly — high MSE = anomaly signal.",
                "tag1": "AUC-ROC: 0.97",
                "tag2": "Threshold: 95th pct",
            },
            {
                "icon": "📉",
                "name": "MSE Separation is Clear",
                "desc": "Normal system metrics produce MSE < 0.008. Anomalous states (memory leak, "
                        "CPU exhaustion, disk I/O spike) produce MSE 8–40× higher on average.",
                "tag1": "Separation: High",
                "tag2": "False Pos.: Low",
            },
            {
                "icon": "🏆",
                "name": "Best Overall Model",
                "desc": "Highest F1 (85%) and AUC-ROC (0.97). Recommended for production deployment "
                        "alongside Isolation Forest as an ensemble for maximum coverage.",
                "tag1": "Precision: 88%",
                "tag2": "Recall: 83%",
            },
        ]
    },
]

effort_colors = {
    "Speed: Real-time":   ("#ECFDF5", "#065F46"),
    "Effort: Low":        ("#F0FDF4", "#166534"),
    "Effort: Medium":     ("#FEF3C7", "#92400E"),
    "Precision: 84%":     ("#EFF6FF", "#1E40AF"),
    "Recall: 79%":        ("#FEF3C7", "#92400E"),
    "Precision: 88%":     ("#EFF6FF", "#1E40AF"),
    "Recall: 83%":        ("#FEF3C7", "#92400E"),
    "Unsupervised: ✓":    ("#ECFDF5", "#065F46"),
    "Scalable: ✓":        ("#ECFDF5", "#065F46"),
    "No score output":    ("#F8FAFC", "#334155"),
    "Hard labels only":   ("#F8FAFC", "#334155"),
    "Sensitivity: Medium":("#FEF3C7", "#92400E"),
    "F1: 73%":            ("#FFFBEB", "#92400E"),
    "AUC-ROC: N/A":       ("#F1F5F9", "#334155"),
    "No label Required":  ("#ECFDF5", "#065F46"),
    "AUC-ROC: 0.97":      ("#F5F3FF", "#5B21B6"),
    "Threshold: 95th pct":("#F8FAFC", "#334155"),
    "Separation: High":   ("#ECFDF5", "#065F46"),
    "False Pos.: Low":    ("#F0FDF4", "#166534"),
    "Precision: 84%":     ("#EFF6FF", "#1E40AF"),
}
default_tag = ("#F1F5F9", "#334155")

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🖥️ System Anomaly Detection Dashboard")
st.markdown("**Data Mining Project** · OS Kernel Anomaly Dataset (Kaggle) · ~1,000 system telemetry records")
st.markdown("---")

c1, c2, c3, c4 = st.columns(4)
c1.metric("ຂໍ້ມູນທັງໝົດ", "1,000 records")
c2.metric("Features", "20 selected")
c3.metric("Best AUC-ROC", "0.97")
c4.metric("Models", "3 Unsupervised")

st.markdown("---")

left, right = st.columns([1, 1.7], gap="large")

with left:
    st.markdown("#### ເລືອກ Detection Model")
    selected = st.radio(
        "Model",
        options=[m["id"] for m in MODELS],
        format_func=lambda x: f"{MODELS[x]['name']}  —  {MODELS[x]['subtitle']}",
        label_visibility="collapsed"
    )
    m = MODELS[selected]

    bars = [("Precision", m["precision"]), ("Recall", m["recall"]), ("F1 Score", m["f1"])]
    if m["auc"] > 0:
        bars.append(("AUC-ROC", m["auc"]))

    bar_html = "".join([
        f'''<div style="display:flex;align-items:center;gap:8px;margin-bottom:7px">
        <div style="font-size:11px;color:#94a3b8;width:80px">{lbl}</div>
        <div style="flex:1;height:6px;background:#f1f5f9;border-radius:4px;overflow:hidden">
            <div style="width:{val}%;height:100%;background:{m["color"]};border-radius:4px"></div>
        </div>
        <div style="font-size:11px;color:{m["color"]};font-weight:700;width:36px;text-align:right">{val}%</div>
        </div>''' for lbl, val in bars
    ])

    st.markdown(f"""
    <div style="margin-top:1rem;background:white;border:1.5px solid {m['color']}33;
         border-top:4px solid {m['color']};border-radius:16px;padding:18px 20px;">
        <div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:3px">{m['name']}</div>
        <div style="font-size:12px;color:#94a3b8;margin-bottom:14px">{m['subtitle']}</div>
        {bar_html}
        <div style="margin-top:12px;font-size:12px;padding:9px 13px;
             background:{m['bgLight']};border-radius:8px;color:{m['textColor']};font-weight:600;">
            ✦ {m['goal']}
        </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown(f"#### Detection Insights — {m['name']}")
    for ins in m["insights"]:
        bg1, tc1 = effort_colors.get(ins["tag1"], default_tag)
        bg2, tc2 = effort_colors.get(ins["tag2"], default_tag)
        st.markdown(f"""
        <div style="display:flex;gap:14px;align-items:flex-start;
             background:#f8fafc;border-radius:12px;padding:16px;margin-bottom:10px;">
            <div style="width:40px;height:40px;border-radius:10px;background:{m['bgLight']};
                 color:{m['textColor']};display:flex;align-items:center;justify-content:center;
                 font-size:20px;flex-shrink:0">{ins['icon']}</div>
            <div style="flex:1">
                <div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:5px">{ins['name']}</div>
                <div style="font-size:13px;color:#475569;line-height:1.65;margin-bottom:9px">{ins['desc']}</div>
                <span style="font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;
                      background:{bg1};color:{tc1};margin-right:6px">{ins['tag1']}</span>
                <span style="font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;
                      background:{bg2};color:{tc2}">{ins['tag2']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center;font-size:12px;color:#cbd5e1;padding:0.5rem 0;'>
    System Anomaly Detection · Kaggle OS Kernel Dataset · Isolation Forest · DBSCAN · Autoencoder
</div>
""", unsafe_allow_html=True)
