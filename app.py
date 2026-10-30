import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="DiabetesAI", page_icon="🩺", layout="wide")

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
MODEL_PATH = Path("diabetes_model.pkl")
CSV_PATH = Path("diabetes.csv")


# ---------- Model ----------
@st.cache_resource
def get_model():
    """Load saved model; otherwise train from diabetes.csv and save it."""
    if MODEL_PATH.exists():
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict):
            return bundle["model"], bundle.get("accuracy")
        return bundle, None
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        x, y = df[FEATURES], df["Outcome"]
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=42)
        m = LogisticRegression(max_iter=1000).fit(x_tr, y_tr)
        acc = float(m.score(x_te, y_te))
        joblib.dump({"model": m, "accuracy": acc}, MODEL_PATH)
        return m, acc
    return None, None


model, accuracy = get_model()
acc_text = f"{accuracy*100:.0f}%" if accuracy else "~75%"

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }
.stApp { background: #f8fafd; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; max-width: 1250px; }

/* Sidebar */
section[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, #0b2a5b 0%, #0a2250 55%, #081a3d 100%);
}
section[data-testid="stSidebar"] * { color: #e8efff; }
.brand { display:flex; align-items:center; gap:10px; margin: 6px 0 22px; }
.brand .logo { font-size: 34px; }
.brand b { font-size: 20px; color:#fff; display:block; line-height:1.1; }
.brand span { font-size: 12px; color:#b9c9ee; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 6px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 12px 16px; border-radius: 10px; width: 100%; cursor: pointer;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child { display:none; }
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #1e6df2; box-shadow: 0 4px 14px rgba(30,109,242,.45);
}
.side-foot { text-align:center; margin-top: 140px; color:#b9c9ee; font-size:16px; line-height:1.4; }

/* Header */
.topbar { text-align:right; color:#475467; font-size:13px; margin-bottom:6px; }
.hero h1 { font-size: 44px; font-weight: 800; color:#0f1b3d; margin:0; line-height:1.1; }
.hero h1 span { color:#1e6df2; }
.hero h3 { font-size: 20px; font-weight: 500; color:#1f2a44; margin: 6px 0 18px; }
.hero p { color:#667085; margin-bottom: 20px; }

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:#fff; border-radius:14px; border:1px solid #e4eaf4 !important;
    box-shadow: 0 2px 10px rgba(16,40,90,.05);
}
.card-title { font-size: 22px; font-weight: 700; color:#0f1b3d; margin:0 0 2px; }
.card-sub { color:#667085; font-size:14px; margin-bottom: 14px; }
div[data-testid="stNumberInput"] label p { font-weight:600; color:#1f2a44; }
div[data-testid="stNumberInput"] input { background:#fbfcff; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg,#1e6df2,#0f57d6); border:0; color:#fff;
    font-weight:600; font-size:16px; padding: 12px 0; border-radius:8px;
}
div.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

/* Result */
.res { padding:22px; border-radius:12px; margin: 12px 0 18px; }
.res.ok  { background:#eefbf3; border:2px solid #7ddba0; }
.res.bad { background:#fef3f2; border:2px solid #f5a5a0; }
.res .head { display:flex; align-items:center; gap:16px; }
.res .icon { width:58px; height:58px; border-radius:50%; display:flex; align-items:center;
             justify-content:center; font-size:30px; color:#fff; }
.res.ok .icon { background:#1fa65a; } .res.bad .icon { background:#e04438; }
.res h2 { margin:0; font-size:30px; font-weight:800; }
.res.ok h2 { color:#0a7a3f; } .res.bad h2 { color:#b42318; }
.res h4 { margin:0; font-size:20px; font-weight:600; }
.res .msg { margin-top:14px; padding:12px 14px; border-radius:8px; font-size:15px; }
.res.ok .msg { background:#d9f4e3; } .res.bad .msg { background:#fde0dd; }
.notice { background:#eaf2ff; color:#1d3a73; padding:14px 16px; border-radius:8px; font-size:14px; }
.empty { color:#667085; padding: 40px 10px; text-align:center; }

/* Info tiles */
.tile h5 { margin:0 0 8px; font-size:17px; font-weight:700; color:#0f1b3d; }
.tile p { margin:2px 0; color:#475467; font-size:13px; }
.tile p.m { color:#98a2b3; }
.footer { text-align:center; color:#667085; font-size:12px; margin-top: 30px; padding-top: 16px;
          border-top: 1px solid #e4eaf4; }
</style>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="brand"><div class="logo">💙</div>'
                '<div><b>DiabetesAI</b><span>Smart Health Prediction</span></div></div>',
                unsafe_allow_html=True)
    page = st.radio("Menu", ["🏠  Home", "📋  Predict", "ℹ️  About", "🧩  Model Info", "✉️  Contact"],
                    label_visibility="collapsed")
    st.markdown('<div class="side-foot">🫀<br>Early Detection<br>Saves Lives</div>',
                unsafe_allow_html=True)

st.markdown('<div class="topbar">👤 ML Project &nbsp;|&nbsp; 🎓 SBBU IT</div>', unsafe_allow_html=True)


def info_tiles():
    tiles = [
        ("🗄️ Dataset", "Pima Indians Diabetes Dataset", "768 samples • 8 features"),
        ("🧠 Algorithm", "Logistic Regression", "Machine Learning Model"),
        ("🎯 Accuracy", acc_text, "(on test data)"),
        ("🛡️ Model Info", "Diabetes Model v1.0", "Trained with Scikit-learn"),
    ]
    cols = st.columns(4)
    for col, (t, a, b) in zip(cols, tiles):
        with col, st.container(border=True):
            st.markdown(f'<div class="tile"><h5>{t}</h5><p>{a}</p><p class="m">{b}</p></div>',
                        unsafe_allow_html=True)


# ---------- Pages ----------
if page.strip().endswith(("Home", "Predict")):
    st.markdown("""
    <div class="hero">
      <h1>Diabetes <span>Prediction</span> System</h1>
      <h3>Machine Learning Based Patient Risk Prediction</h3>
      <p>Enter the patient information below to check the likelihood of diabetes.</p>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")

    with left, st.container(border=True):
        st.markdown('<div class="card-title">👤 Patient Information</div>'
                    '<div class="card-sub">Fill in the following details to get the prediction.</div>',
                    unsafe_allow_html=True)
        a, b = st.columns(2)
        preg = a.number_input("Pregnancies", 0, 20, 1)
        ins = b.number_input("Insulin", 0, 900, 80)
        glu = a.number_input("Glucose", 0, 300, 120)
        bmi = b.number_input("BMI", 0.0, 80.0, 25.0, step=0.1)
        bp = a.number_input("Blood Pressure", 0, 200, 70)
        dpf = b.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5, step=0.01)
        skin = a.number_input("Skin Thickness", 0, 100, 20)
        age = b.number_input("Age", 1, 120, 30)
        go = st.button("🔍  Predict Diabetes", type="primary", width="stretch")

    with right, st.container(border=True):
        st.markdown('<div class="card-title">📊 Prediction Result</div>', unsafe_allow_html=True)

        if go and model is None:
            st.error("Model nahi mila. `diabetes.csv` ko app.py ke folder mein rakhein.")
        elif go:
            row = pd.DataFrame([[preg, glu, bp, skin, ins, bmi, dpf, age]], columns=FEATURES)
            pred = int(model.predict(row)[0])
            prob = float(model.predict_proba(row)[0][1]) * 100
            if pred == 0:
                st.markdown(f"""<div class="res ok"><div class="head"><div class="icon">✓</div>
                  <div><h2>Diabetes = NO</h2><h4>(Not Detected)</h4></div></div>
                  <div class="msg">Based on the provided data, the model predicts that the patient is
                  <b>not likely</b> to have diabetes. (Risk probability: {prob:.1f}%)</div></div>""",
                            unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="res bad"><div class="head"><div class="icon">!</div>
                  <div><h2>Diabetes = YES</h2><h4>(Detected)</h4></div></div>
                  <div class="msg">Based on the provided data, the model predicts that the patient is
                  <b>likely</b> to have diabetes. (Risk probability: {prob:.1f}%)</div></div>""",
                            unsafe_allow_html=True)

            st.markdown("**👤 Input Summary**")
            s = st.columns(2)
            items = [("Pregnancies", preg), ("Glucose", glu), ("Blood Pressure", bp),
                     ("Skin Thickness", skin), ("Insulin", ins), ("BMI", bmi),
                     ("Pedigree Function", dpf), ("Age", age)]
            s[0].dataframe(pd.DataFrame(items[:4], columns=["Feature", "Value"]).astype(str),
                           hide_index=True, width="stretch")
            s[1].dataframe(pd.DataFrame(items[4:], columns=["Feature", "Value"]).astype(str),
                           hide_index=True, width="stretch")
        else:
            st.markdown('<div class="empty">Patient ki details bharein aur '
                        '<b>Predict Diabetes</b> dabayein.</div>', unsafe_allow_html=True)

        st.markdown('<div class="notice">ℹ️ This prediction is based on a machine learning model '
                    'and is not a substitute for professional medical diagnosis.</div>',
                    unsafe_allow_html=True)

    st.write("")
    info_tiles()

elif "About" in page:
    st.title("About")
    st.write("DiabetesAI ek machine learning project hai jo Pima Indians Diabetes Dataset par "
             "train kiye gaye Logistic Regression model se patient ka diabetes risk predict karta hai.")
    st.info("Yeh sirf educational use ke liye hai, medical diagnosis nahi.")

elif "Model Info" in page:
    st.title("Model Info")
    info_tiles()
    st.write("")
    st.write("**Features:** " + ", ".join(FEATURES))

else:
    st.title("Contact")
    st.write("Developed by **Razique Brohi** — SBBU IT, Machine Learning Project.")

st.markdown('<div class="footer">Diabetes Prediction System &nbsp;|&nbsp; Machine Learning Project '
            '&nbsp;|&nbsp; Developed by Razique Brohi</div>', unsafe_allow_html=True)
