import streamlit as st
import pandas as pd

# --- ตั้งค่าหน้าเว็บ ---
st.set_page_config(page_title="NeuroLNP AI Screener", layout="wide")

st.title("🧠 NeuroLNP Platform: AI-Powered BBB Penetration Screener")
st.markdown("**แพลตฟอร์มคัดกรองโครงสร้าง Lipid Nanoparticle (LNP) มุ่งเป้าสมอง สำหรับรักษาโรคอัลไซเมอร์**")
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("🧪 สารประกอบที่ AI คัดกรอง", "150+ สูตร", "Generative Library")
col2.metric("🏆 สูตรที่ได้คะแนนสูงสุด (logBB)", "0.2918", "LNP_Unsat_C12_Linker2")
col3.metric("🧬 ความเสถียรในการจับตัวรับ (Vina Score)", "-6.1 kcal/mol", "Acetylcholine Receptor")

st.markdown("---")

st.subheader("📊 ผลการคัดกรอง Top 5 LNP Candidates (Virtual Screening)")
data = {
    "อันดับ": [1, 2, 3, 4, 5],
    "ชื่อสูตร (Formula Name)": ["LNP_Unsat_C12_Linker2", "LNP_Sat_C9_Linker4", "LNP_Sat_C8_Linker5", "LNP_Unsat_C11_Linker3", "LNP_Sat_C10_Linker4"],
    "รหัสโครงสร้าง (SMILES)": ["CCCCCCCCC=CCCC(=O)OCC[N+](C)(C)C", "CCCCCCCCCC(=O)OCCCC[N+](C)(C)C", "CCCCCCCCC(=O)OCCCCC[N+](C)(C)C", "CCCCCCCC=CCCC(=O)OCCC[N+](C)(C)C", "CCCCCCCCCCC(=O)OCCCC[N+](C)(C)C"],
    "คะแนนการเข้าสมอง (Predicted logBB)": ["0.2918", "0.2705", "0.2705", "0.2691", "0.2658"]
}
df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True)

st.subheader("🔬 การยืนยันผลด้วย Molecular Docking (Targeting Acetylcholine Receptor)")
st.info("💡 นำสูตรอันดับ 1 เข้าจำลองการจับตัวกับโปรตีนสมอง 2XZ5 พบว่าส่วนหัวของ LNP สามารถเข้าไปในรูกุญแจ (Cavity) ได้อย่างแม่นยำ ด้วยพลังงานการยึดเหนี่ยวที่ -6.1 kcal/mol")

# โค้ดสำหรับแสดงรูปภาพ 3D
try:
    st.image("docking_result.png", caption="ภาพจำลอง 3 มิติ: การเกาะติดของ LNP_Unsat_C12_Linker2 กับตัวรับ Acetylcholine (2XZ5)")
except:
    st.warning("⚠️ ไม่พบไฟล์รูปภาพ 'docking_result.png' กรุณาเพิ่มรูปภาพลงในโฟลเดอร์")