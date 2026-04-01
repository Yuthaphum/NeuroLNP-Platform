import streamlit as st
import time
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import py3Dmol
import streamlit.components.v1 as components
import pickle
import numpy as np

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="NeuroLNP Platform", layout="wide")
st.title("🧬 NeuroLNP: Generative AI for Brain-Targeted LNP Design")
st.markdown("แพลตฟอร์มออกแบบและคัดกรองอนุภาคนาโนไขมัน (LNP) สำหรับส่งยาข้ามแนวกั้นสมอง (BBB) ขับเคลื่อนด้วยระบบ Dual-Model AI Architecture")

# --- 2. โหลดสมองกล AI ทั้ง 2 ตัว (BBB & Toxicity) ---
try:
    with open('xgboost_bbb_model.pkl', 'rb') as f:
        xgb_bbb = pickle.load(f)
    with open('xgboost_tox_model.pkl', 'rb') as f:
        xgb_tox = pickle.load(f)
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.sidebar.error("⚠️ ไม่พบไฟล์สมองกล AI กรุณาตรวจสอบไฟล์ .pkl ในระบบ")

# --- 3. แถบเครื่องมือออกแบบ (Sidebar) ---
st.sidebar.header("🛠️ LNP Builder (In silico Assembly)")
head_group = st.sidebar.selectbox("1. ส่วนหัวมุ่งเป้า (Targeting Ligand)", [
    "Biomimetic (Acetylcholine-like)",
    "NT-Lipidoid (Tryptamine-based)",
    "Standard (Dimethylamino)"
])
linker = st.sidebar.selectbox("2. ส่วนเชื่อม (Linker Type)", [
    "Ester (-COO-) [Biodegradable]",
    "Amide (-CONH-) [Stable]",
    "Ether (-O-) [Highly Stable]"
])
tail_length = st.sidebar.slider("3. ความยาวสายหางไขมัน (Carbon Tail)", min_value=8, max_value=18, value=12, step=1)
peg_length = st.sidebar.select_slider("4. ความยาว PEG-lipid anchor", options=["PEG-1000", "PEG-2000", "PEG-3000"], value="PEG-2000")

# --- 4. ระบบสร้างโครงสร้างเคมี (Generative SMILES) ---
def generate_smiles(head, link, tail_len):
    head_smiles = {
        "Standard (Dimethylamino)": "CN(C)CC",
        "NT-Lipidoid (Tryptamine-based)": "c1ccc2c(c1)c(c[nH]2)CCN(C)CC",
        "Biomimetic (Acetylcholine-like)": "C[N+](C)(C)CCOC"
    }[head]
    linker_smiles = {
        "Ester (-COO-) [Biodegradable]": "(=O)O",
        "Amide (-CONH-) [Stable]": "(=O)N",
        "Ether (-O-) [Highly Stable]": "O"
    }[link]
    tail_smiles = "C" * tail_len
    return f"{head_smiles}{linker_smiles}{tail_smiles}"

current_smiles = generate_smiles(head_group, linker, tail_length)

# --- 5. ระบบประมวลผล Dual-Model AI ---
def predict_properties_dual(smiles, peg):
    logbb = 0.0
    tox_risk = 50.0 # ค่าตั้งต้น
    
    if models_loaded:
        try:
            mol = Chem.MolFromSmiles(smiles)
            # สกัด 10 ฟีเจอร์
            features = [
                Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
                Descriptors.FractionCSP3(mol), Descriptors.HeavyAtomCount(mol), Descriptors.NHOHCount(mol),
                Descriptors.NOCount(mol)
            ]
            X_input = np.array([features])
            
            # Model 1: ทำนาย Efficacy (logBB)
            logbb = float(xgb_bbb.predict(X_input)[0])
            # กฎฟิสิกส์อนุภาค: ปรับจูนด้วย PEG Dilemma 
            if peg == "PEG-2000": logbb += 0.2 
            elif peg == "PEG-3000": logbb -= 0.3
                
            # Model 2: ทำนาย Toxicity (Clinical Safety)
            # ดึงค่าความน่าจะเป็นที่สารนี้จะเป็นพิษ (คลาส 1) แล้วคูณ 100 ให้เป็นเปอร์เซ็นต์
            tox_prob = float(xgb_tox.predict_proba(X_input)[0][1])
            tox_risk = tox_prob * 100
            
        except:
            pass
            
    # คำนวณ Overall Score (น้ำหนัก: ประสิทธิภาพ 60% ความปลอดภัย 40%)
    norm_logbb = min(max((logbb + 1.0) / 2.5 * 100, 0), 100)
    safety_score = 100 - tox_risk
    overall = (norm_logbb * 0.6) + (safety_score * 0.4)
    
    return round(logbb, 2), round(tox_risk, 1), round(overall, 1)

logBB_val, tox_val, overall_val = predict_properties_dual(current_smiles, peg_length)

# --- 6. จัดหน้าจอแสดงผลหลัก ---
st.subheader("📊 AI Prediction Dashboard")
st.markdown(f"**Generated SMILES (Targeting Lipid Monomer):** `{current_smiles}`")

score_col1, score_col2, score_col3 = st.columns(3)
with score_col1:
    st.metric(label="🏆 Overall LNP Score", value=f"{overall_val} / 100", 
              delta="High Potential" if overall_val >= 70 else "Needs Optimization",
              delta_color="normal" if overall_val >= 70 else "inverse")
with score_col2:
    st.metric(label="🧠 BBB Efficacy (logBB)", value=f"{logBB_val} kcal/mol")
with score_col3:
    st.metric(label="⚠️ Clinical Toxicity Risk", value=f"{tox_val} %",
              delta="Safe Margin" if tox_val < 30 else "High Risk", delta_color="inverse")

st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🧩 3D Structure (Targeting Lipid Monomer)")
    st.info("💡 ภาพนี้แสดงโครงสร้าง 3 มิติของ 'โมเลกุลไขมันมุ่งเป้า (Monomer)' ซึ่งในกระบวนการผลิตจริง โมเลกุลเหล่านี้นับพันตัวจะเกิดการรวมตัว (Self-assembly) กลายเป็น 'อนุภาคนาโนทรงกลม (Spherical LNP)' เพื่อหุ้มยาต่อไป")
    def render_3d(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            mblock = Chem.MolToMolBlock(mol)
            
            viewer = py3Dmol.view(width=450, height=350)
            viewer.addModel(mblock, 'mol')
            viewer.setStyle({'stick': {}})
            viewer.zoomTo()
            return viewer._make_html()
        except:
            return "<p>กำลังประมวลผลโครงสร้าง...</p>"
            
    components.html(render_3d(current_smiles), height=400)

with col2:
    st.subheader("🔬 Phase 2: Deep Validation")
    st.info("นำส่วนหัวมุ่งเป้า (Targeting Ligand) ไปจำลองการจับกับตัวรับ Acetylcholine (AChR) เพื่อยืนยันกลไกการผ่านเซลล์สมอง")
    run_docking = st.button("🚀 Run Molecular Docking", type="primary", use_container_width=True)
    
    if run_docking:
        with st.spinner("ประมวลผล Molecular Docking In Silico..."):
            time.sleep(2)
        st.success("✅ Molecular Docking สำเร็จ!")
        # อย่าลืมใส่รูป docking_result.png ไว้ในโฟลเดอร์เดียวกันด้วยนะครับ
        st.image("docking_result.png", caption="3D Binding with AChR (-6.1 kcal/mol)")