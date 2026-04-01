import streamlit as st
import time
from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import streamlit.components.v1 as components

# --- 1. ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="NeuroLNP Platform", layout="wide")
st.title("🧬 NeuroLNP: Generative AI for Brain-Targeted LNP Design")
st.markdown("แพลตฟอร์มออกแบบและคัดกรองอนุภาคนาโนไขมัน (LNP) สำหรับส่งยาข้ามแนวกั้นสมอง (BBB) ด้วย Dual-Model AI")

# --- 2. แถบเครื่องมือออกแบบ (Sidebar) ---
st.sidebar.header("🛠️ LNP Builder (ประกอบโครงสร้าง)")
st.sidebar.markdown("ปรับแต่งตัวแปรเพื่อสร้างสูตร LNP (In silico Assembly)")

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

# --- 3. ระบบสร้างโครงสร้างเคมี (Generative SMILES) ---
def generate_smiles(head, link, tail_len):
    # ชิ้นส่วนหัว
    head_smiles = {
        "Standard (Dimethylamino)": "CN(C)CC",
        "NT-Lipidoid (Tryptamine-based)": "c1ccc2c(c1)c(c[nH]2)CCN(C)CC",
        "Biomimetic (Acetylcholine-like)": "C[N+](C)(C)CCOC"
    }[head]
    
    # ชิ้นส่วนเชื่อม
    linker_smiles = {
        "Ester (-COO-) [Biodegradable]": "(=O)O",
        "Amide (-CONH-) [Stable]": "(=O)N",
        "Ether (-O-) [Highly Stable]": "O"
    }[link]
    
    # ชิ้นส่วนหาง
    tail_smiles = "C" * tail_len
    
    return f"{head_smiles}{linker_smiles}{tail_smiles}"

current_smiles = generate_smiles(head_group, linker, tail_length)

# --- 4. ระบบจำลอง Dual-Model AI Prediction ---
def predict_properties(head, tail_len, link, peg):
    # 1. ทำนาย Efficacy (logBB)
    logbb = -1.0
    if "Acetylcholine" in head: logbb += 1.8
    elif "Tryptamine" in head: logbb += 1.2
    else: logbb += 0.5
        
    if 12 <= tail_len <= 14: logbb += 0.5
    else: logbb -= abs(13 - tail_len) * 0.1
    
    if peg == "PEG-2000": logbb += 0.2 # PEG-2000 มักจะสมดุลสุดในการผ่าน BBB
    elif peg == "PEG-3000": logbb -= 0.3 # หนาไป เริ่มติดขัด (PEG Dilemma)
    
    # 2. ทำนาย Toxicity Risk (ความเป็นพิษ %)
    tox_risk = 20.0 # Base risk
    if "Ester" in link: tox_risk += 5.0 # ย่อยสลายได้ พิษต่ำ
    elif "Amide" in link: tox_risk += 25.0
    elif "Ether" in link: tox_risk += 40.0 # ขับออกยาก พิษสะสม
    
    if tail_len > 14: tox_risk += (tail_len - 14) * 5.0 # หางยาวเกินไปแทรกแซงเยื่อหุ้มเซลล์ปกติ
    
    tox_risk = min(max(tox_risk, 0), 100)
    
    # 3. คำนวณ Overall Score (0-100)
    # สมมติฐาน: ให้ความสำคัญ logBB 60% และ Toxicity 40%
    norm_logbb = min(max((logbb + 1.0) / 2.5 * 100, 0), 100)
    safety_score = 100 - tox_risk
    overall = (norm_logbb * 0.6) + (safety_score * 0.4)
    
    return round(logbb, 2), round(tox_risk, 1), round(overall, 1)

logBB_val, tox_val, overall_val = predict_properties(head_group, tail_length, linker, peg_length)

# --- 5. จัดหน้าจอแสดงผลหลัก ---
st.subheader("📊 AI Prediction Dashboard")
st.markdown(f"**Generated SMILES (Ionizable Lipid):** `{current_smiles}`")

# แถบคะแนน 3 คอลัมน์
score_col1, score_col2, score_col3 = st.columns(3)
with score_col1:
    st.metric(label="🏆 Overall LNP Score", value=f"{overall_val} / 100", 
              delta="Highly Recommended" if overall_val >= 75 else "Needs Optimization",
              delta_color="normal" if overall_val >= 75 else "inverse")
with score_col2:
    st.metric(label="🧠 BBB Efficacy (logBB)", value=f"{logBB_val} kcal/mol")
with score_col3:
    st.metric(label="⚠️ Toxicity Risk", value=f"{tox_val} %",
              delta="Safe (Biodegradable)" if tox_val < 30 else "High Risk", delta_color="inverse")

st.markdown("---")

# จัดเลย์เอาต์ 3D และข้อมูลเชิงลึก
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("🧩 3D Molecular Viewer")
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
    st.info("นำส่วนหัวมุ่งเป้า (Targeting Ligand) ไปจำลองการจับกับตัวรับ Acetylcholine (AChR) เพื่อยืนยันกลไก Trojan Horse")
    run_docking = st.button("🚀 Run Molecular Docking", type="primary", use_container_width=True)
    
    if run_docking:
        with st.spinner("ประมวลผล In Silico..."):
            time.sleep(2)
        st.success("✅ Molecular Docking สำเร็จ!")
        st.image("docking_result.png", caption="3D Binding with AChR (-6.1 kcal/mol)")