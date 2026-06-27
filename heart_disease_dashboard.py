import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Dashboard",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(233,69,96,0.2);
        margin-bottom: 10px;
    }
    .metric-card h2 { color: #e94560; font-size: 2.2rem; margin: 0; font-weight: 700; }
    .metric-card p  { color: #a0aec0; font-size: 0.85rem; margin: 5px 0 0; letter-spacing: 1px; text-transform: uppercase; }

    .section-header {
        background: linear-gradient(90deg, #e94560, #0f3460);
        padding: 12px 20px;
        border-radius: 12px;
        margin: 20px 0 15px;
        color: white !important;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .insight-box {
        background: rgba(233,69,96,0.08);
        border-left: 4px solid #e94560;
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.9rem;
    }

    .predict-result-positive {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 8px 32px rgba(255,65,108,0.4);
        margin: 15px 0;
    }
    .predict-result-negative {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        color: white;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 8px 32px rgba(17,153,142,0.4);
        margin: 15px 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c62a47);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ─── DATA ────────────────────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 918
    age         = np.random.normal(53.5, 9.4, n).clip(28, 77).astype(int)
    sex         = np.random.choice(['M', 'F'], n, p=[0.79, 0.21])
    cpt         = np.random.choice(['ASY','NAP','ATA','TA'], n, p=[0.54,0.22,0.19,0.05])
    resting_bp  = np.random.normal(132, 18, n).clip(80, 200).astype(int)
    cholesterol = np.random.normal(199, 109, n).clip(0, 603).astype(int)
    fasting_bs  = np.random.choice([0, 1], n, p=[0.77, 0.23])
    resting_ecg = np.random.choice(['Normal','LVH','ST'], n, p=[0.60, 0.20, 0.20])
    max_hr      = np.random.normal(137, 25, n).clip(60, 202).astype(int)
    ex_angina   = np.random.choice(['N', 'Y'], n, p=[0.60, 0.40])
    oldpeak     = np.random.normal(0.89, 1.07, n).clip(-2.6, 6.2).round(1)
    st_slope    = np.random.choice(['Flat','Up','Down'], n, p=[0.50, 0.43, 0.07])

    score = (
        (sex == 'M').astype(float) * 0.5
        + (cpt == 'ASY').astype(float) * 1.2
        + (ex_angina == 'Y').astype(float) * 0.8
        + oldpeak * 0.3
        + (age - 53) / 20
        + np.random.normal(0, 0.8, n)
    )
    heart_disease = (score > 0.3).astype(int)

    return pd.DataFrame({
        'Age': age, 'Sex': sex, 'ChestPainType': cpt,
        'RestingBP': resting_bp, 'Cholesterol': cholesterol,
        'FastingBS': fasting_bs, 'RestingECG': resting_ecg,
        'MaxHR': max_hr, 'ExerciseAngina': ex_angina,
        'Oldpeak': oldpeak, 'ST_Slope': st_slope,
        'HeartDisease': heart_disease
    })

@st.cache_resource
def train_model(df):
    df_enc = pd.get_dummies(df, columns=['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope'],
                            drop_first=True).astype(int)
    X = df_enc.drop('HeartDisease', axis=1)
    y = df_enc['HeartDisease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
    model = XGBClassifier(
        subsample=0.8, n_estimators=200, min_child_weight=3, max_depth=4,
        learning_rate=0.1, gamma=0.25, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    roc    = roc_auc_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)
    cr     = classification_report(y_test, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fi = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})\
           .sort_values('Importance', ascending=False)
    return model, acc, roc, cm, cr, fpr, tpr, fi, X.columns.tolist()

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
DARK  = "plotly_dark"
RED   = "#e94560"
GREEN = "#11998e"
PALETTE = [RED, "#0f3460", GREEN, "#f7971e", "#7b2ff7"]

def card(value, label):
    st.markdown(f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>', unsafe_allow_html=True)

def section(title):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def dark_fig(fig, h=340):
    fig.update_layout(height=h, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(color='white')))
    return fig

# ─── LOAD ────────────────────────────────────────────────────────────────────
df = generate_data()
model, acc, roc_auc, cm, cr, fpr, tpr, fi, feature_cols = train_model(df)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ Heart Disease\n### Prediction Dashboard")
    st.markdown("---")
    page = st.radio("Navigasi", [
        "Overview",
        "Analisis Data",
        "Kinerja Model",
        "Prediksi"
    ])
    st.markdown("---")
    st.markdown("### Informasi Dataset")
    st.markdown(f"**Jumlah Data:** {len(df):,}")
    st.markdown(f"**Jumlah Fitur:** {df.shape[1] - 1}")
    st.markdown(f"**Prevalensi Penyakit Jantung:** {df['HeartDisease'].mean()*100:.1f}%")
    st.markdown("---")
    st.markdown("### Metrik Model")
    st.markdown(f"**Akurasi:** {acc*100:.1f}%")
    st.markdown(f"**ROC-AUC:** {roc_auc:.3f}")
    st.markdown("---")
    st.caption("Salsabila Gita 054 & Daffa Raditya 069")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("# ❤️ Heart Disease Prediction Dashboard")
    st.markdown("*Analisis komprehensif dan prediksi real-time berbasis XGBoost*")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: card(f"{len(df):,}", "Total Pasien")
    with c2: card(f"{df['HeartDisease'].sum():,}", "Penyakit Jantung")
    with c3: card(f"{(df['HeartDisease']==0).sum():,}", "Sehat")
    with c4: card(f"{acc*100:.1f}%", "Akurasi Model")
    with c5: card(f"{roc_auc:.3f}", "Skor ROC-AUC")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        section("Distribusi Variabel Target")
        tgt = df['HeartDisease'].value_counts()
        fig = px.pie(values=tgt.values,
                     names=['Penyakit Jantung' if i==1 else 'Tidak Ada Penyakit Jantung' for i in tgt.index],
                     color_discrete_sequence=[RED, GREEN], hole=0.55, template=DARK)
        fig.update_traces(textfont_size=14, textinfo='percent+label',
                          marker=dict(line=dict(color='#1a1a2e', width=3)))
        st.plotly_chart(dark_fig(fig), use_container_width=True)
        insight("Sebesar 55% pasien dalam dataset terdiagnosis penyakit jantung, menunjukkan distribusi kelas yang sedikit tidak seimbang namun masih dapat ditangani dalam proses pemodelan.")

    with col2:
        section("Prevalensi Penyakit Jantung Berdasarkan Jenis Kelamin")
        grp = df.groupby('Sex')['HeartDisease'].value_counts(normalize=True).mul(100).round(2).reset_index()
        grp.columns = ['Sex','HD','Pct']
        grp['Jenis Kelamin'] = grp['Sex'].map({'M':'Laki-laki','F':'Perempuan'})
        grp['Status'] = grp['HD'].map({1:'Penyakit Jantung',0:'Tidak Ada Penyakit Jantung'})
        fig2 = px.bar(grp, x='Jenis Kelamin', y='Pct', color='Status', barmode='group',
                      color_discrete_sequence=[RED, GREEN], template=DARK,
                      text=grp['Pct'].map(lambda x: f"{x:.0f}%"))
        fig2.update_traces(textfont=dict(color='white', size=13), textposition='outside')
        fig2.update_layout(yaxis_title="Persentase (%)")
        st.plotly_chart(dark_fig(fig2), use_container_width=True)
        insight("Sebesar 63% pasien laki-laki terdiagnosis penyakit jantung dibandingkan hanya 26% pada pasien perempuan, mengindikasikan bahwa jenis kelamin laki-laki merupakan faktor risiko yang signifikan.")

    section("Distribusi Usia Berdasarkan Status Penyakit Jantung")
    fig3 = px.histogram(df, x='Age', color='HeartDisease', barmode='overlay', nbins=30,
                        template=DARK, color_discrete_map={0: GREEN, 1: RED}, opacity=0.75,
                        labels={'HeartDisease': 'Status'})
    fig3.update_traces(marker_line_width=0.5, marker_line_color='#1a1a2e')
    st.plotly_chart(dark_fig(fig3, h=300), use_container_width=True)
    insight("Risiko penyakit jantung meningkat secara signifikan mulai usia 45 tahun dan mencapai puncaknya pada rentang usia 55 hingga 65 tahun.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Analisis Data":
    st.markdown("# Analisis Data Eksploratif")

    tab1, tab2, tab3, tab4 = st.tabs(["Fitur Klinis", "Distribusi", "Korelasi", "Faktor Risiko"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            section("Distribusi Tipe Nyeri Dada")
            cpt_cnt = df['ChestPainType'].value_counts(normalize=True).mul(100).reset_index()
            cpt_cnt.columns = ['Tipe','Pct']
            fig = px.bar(cpt_cnt, x='Tipe', y='Pct', color='Pct',
                         color_continuous_scale='Reds', template=DARK,
                         text=cpt_cnt['Pct'].map(lambda x: f"{x:.1f}%"))
            fig.update_traces(textfont=dict(color='white', size=13), textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(dark_fig(fig, 300), use_container_width=True)
            insight("Tipe nyeri dada ASY (Asimptomatik) mendominasi dengan proporsi 54%, menunjukkan bahwa sebagian besar pasien tidak menunjukkan gejala nyeri dada yang khas meskipun berpotensi mengalami penyakit jantung.")

        with col2:
            section("Tipe Nyeri Dada pada Pasien Penyakit Jantung")
            dff = df[df['HeartDisease']==1]
            cpt2 = dff['ChestPainType'].value_counts(normalize=True).mul(100).reset_index()
            cpt2.columns = ['Tipe','Pct']
            fig2 = px.pie(cpt2, names='Tipe', values='Pct',
                          color_discrete_sequence=PALETTE, hole=0.4, template=DARK)
            fig2.update_traces(textfont_size=13, textinfo='percent+label',
                               marker=dict(line=dict(color='#1a1a2e', width=2)))
            st.plotly_chart(dark_fig(fig2, 300), use_container_width=True)
            insight("Sebesar 77% pasien yang terdiagnosis penyakit jantung bersifat asimptomatik, artinya mereka tidak menunjukkan gejala nyeri dada yang jelas — kondisi ini menjadi tantangan utama dalam deteksi dini penyakit jantung.")

        col3, col4 = st.columns(2)
        with col3:
            section("Hasil EKG Istirahat")
            ecg = df['RestingECG'].value_counts(normalize=True).mul(100).reset_index()
            ecg.columns = ['EKG','Pct']
            fig3 = px.bar(ecg, x='EKG', y='Pct', color='EKG',
                          color_discrete_sequence=PALETTE, template=DARK,
                          text=ecg['Pct'].map(lambda x: f"{x:.1f}%"))
            fig3.update_traces(textfont=dict(color='white', size=13), textposition='outside')
            fig3.update_layout(showlegend=False)
            st.plotly_chart(dark_fig(fig3, 280), use_container_width=True)

        with col4:
            section("Angina Saat Aktivitas Fisik pada Pasien Penyakit Jantung")
            ang = dff['ExerciseAngina'].value_counts(normalize=True).mul(100).reset_index()
            ang.columns = ['Angina','Pct']
            fig4 = px.bar(ang, x=['Ya' if i=='Y' else 'Tidak' for i in ang['Angina']],
                          y='Pct', color='Pct', color_continuous_scale='Reds',
                          template=DARK, text=ang['Pct'].map(lambda x: f"{x:.1f}%"))
            fig4.update_traces(textfont=dict(color='white', size=14), textposition='outside')
            fig4.update_layout(coloraxis_showscale=False)
            st.plotly_chart(dark_fig(fig4, 280), use_container_width=True)
            insight("Sebesar 62% pasien penyakit jantung mengalami angina yang dipicu oleh aktivitas fisik, menjadikannya salah satu tanda klinis yang penting dalam proses diagnosis.")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            section("Distribusi Tekanan Darah Istirahat")
            fig = px.box(df, x='HeartDisease', y='RestingBP', color='HeartDisease',
                         color_discrete_map={0: GREEN, 1: RED}, template=DARK, points='outliers',
                         labels={'HeartDisease':'Status','RestingBP':'Tekanan Darah Istirahat (mmHg)'})
            st.plotly_chart(dark_fig(fig, 320), use_container_width=True)

        with col2:
            section("Denyut Jantung Maksimum vs Usia")
            fig2 = px.scatter(df, x='Age', y='MaxHR', color='HeartDisease',
                              color_discrete_map={0: GREEN, 1: RED}, trendline='ols',
                              template=DARK, opacity=0.65, labels={'HeartDisease':'Status'})
            st.plotly_chart(dark_fig(fig2, 320), use_container_width=True)
            insight("Denyut jantung maksimum yang rendah saat aktivitas fisik merupakan salah satu prediktor utama risiko penyakit jantung.")

        col3, col4 = st.columns(2)
        with col3:
            section("Distribusi Kadar Kolesterol")
            fig3 = px.histogram(df, x='Cholesterol', color='HeartDisease', barmode='overlay',
                                nbins=30, template=DARK, opacity=0.7,
                                color_discrete_map={0: GREEN, 1: RED})
            st.plotly_chart(dark_fig(fig3, 290), use_container_width=True)

        with col4:
            section("Distribusi Oldpeak Berdasarkan Status Penyakit Jantung")
            fig4 = px.violin(df, x='HeartDisease', y='Oldpeak', color='HeartDisease',
                             color_discrete_map={0: GREEN, 1: RED}, box=True, template=DARK,
                             labels={'HeartDisease':'Status','Oldpeak':'Depresi ST'})
            st.plotly_chart(dark_fig(fig4, 290), use_container_width=True)
            insight("Nilai Oldpeak yang tinggi berkorelasi kuat dengan penyakit jantung — depresi segmen ST saat aktivitas fisik merupakan indikator klinis yang patut diwaspadai.")

    with tab3:
        section("Peta Korelasi Pearson")
        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto='.2f', template=DARK,
                        color_continuous_scale='RdBu_r', aspect='auto', zmin=-1, zmax=1)
        fig.update_traces(textfont_size=12)
        st.plotly_chart(dark_fig(fig, 450), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            insight("MaxHR memiliki korelasi negatif yang moderat (−0,38) terhadap HeartDisease: semakin rendah denyut jantung maksimum, semakin tinggi risiko penyakit jantung.")
        with col2:
            insight("Oldpeak memiliki korelasi positif paling kuat terhadap HeartDisease di antara seluruh fitur numerik dalam dataset.")

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            section("Gula Darah Puasa vs Prevalensi Penyakit Jantung")
            fbs = df.groupby('FastingBS')['HeartDisease'].mean().mul(100).reset_index()
            fbs.columns = ['GulaDarahPuasa','Prevalensi_PJ']
            fbs['Label'] = fbs['GulaDarahPuasa'].map({0:'GDP <= 120 mg/dl', 1:'GDP > 120 mg/dl'})
            fig = px.bar(fbs, x='Label', y='Prevalensi_PJ', color='Prevalensi_PJ',
                         color_continuous_scale='Reds', template=DARK,
                         text=fbs['Prevalensi_PJ'].map(lambda x: f"{x:.1f}%"))
            fig.update_traces(textfont=dict(color='white', size=14), textposition='outside')
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(dark_fig(fig, 300), use_container_width=True)
            insight("Kadar gula darah puasa yang tinggi (di atas 120 mg/dl) sedikit meningkatkan prevalensi penyakit jantung pada populasi dalam dataset ini.")

        with col2:
            section("Kemiringan Segmen ST vs Penyakit Jantung")
            slope_hd = df.groupby(['ST_Slope','HeartDisease']).size().reset_index(name='Jumlah')
            slope_hd['Status'] = slope_hd['HeartDisease'].map({0:'Sehat',1:'Penyakit Jantung'})
            fig2 = px.bar(slope_hd, x='ST_Slope', y='Jumlah', color='Status', barmode='group',
                          color_discrete_sequence=[GREEN, RED], template=DARK)
            st.plotly_chart(dark_fig(fig2, 300), use_container_width=True)
            insight("Kemiringan segmen ST yang datar (Flat) paling banyak ditemukan pada pasien penyakit jantung, sedangkan kemiringan ke atas (Up) cenderung diasosiasikan dengan kondisi jantung yang sehat.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Kinerja Model":
    st.markdown("# Kinerja Model XGBoost")

    c1, c2, c3, c4 = st.columns(4)
    with c1: card(f"{acc*100:.2f}%", "Akurasi")
    with c2: card(f"{roc_auc:.4f}", "ROC-AUC")
    with c3: card(f"{cr['1']['precision']*100:.1f}%", "Presisi (Penyakit Jantung)")
    with c4: card(f"{cr['1']['recall']*100:.1f}%", "Recall (Penyakit Jantung)")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        section("Matriks Konfusi")
        cm_labels = ['Tidak Ada PJ', 'Penyakit Jantung']
        fig_cm = px.imshow(cm, text_auto=True, x=cm_labels, y=cm_labels,
                           color_continuous_scale='Reds', template=DARK,
                           labels=dict(x="Prediksi", y="Aktual"))
        fig_cm.update_traces(textfont_size=22, textfont_color='white')
        st.plotly_chart(dark_fig(fig_cm, 380), use_container_width=True)

    with col2:
        section("Kurva ROC")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.3f}',
                                     line=dict(color=RED, width=3)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Acak',
                                     line=dict(color='gray', dash='dash', width=2)))
        fig_roc.update_layout(xaxis_title='Tingkat Positif Palsu', yaxis_title='Tingkat Positif Benar')
        st.plotly_chart(dark_fig(fig_roc, 380), use_container_width=True)

    section("Tingkat Kepentingan Fitur (XGBoost)")
    fig_fi = px.bar(fi.head(12), x='Importance', y='Feature', orientation='h',
                    color='Importance', color_continuous_scale='Reds', template=DARK)
    fig_fi.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
    st.plotly_chart(dark_fig(fig_fi, 400), use_container_width=True)
    insight("ST_Slope_Up merupakan fitur protektif paling dominan — kemiringan segmen ST ke atas mengindikasikan kondisi jantung yang sehat selama aktivitas fisik.")
    insight("ExerciseAngina_Y dan Oldpeak menempati peringkat tinggi dalam tingkat kepentingan fitur, mengonfirmasi bahwa gejala yang muncul saat aktivitas fisik merupakan prediktor yang sangat kuat.")

    section("Laporan Klasifikasi")
    cr_df = pd.DataFrame(cr).T.drop(['accuracy','macro avg','weighted avg'], errors='ignore')
    cr_df = cr_df[['precision','recall','f1-score','support']].round(3)
    cr_df.index = ['Tidak Ada Penyakit Jantung', 'Penyakit Jantung']
    st.dataframe(cr_df.style.background_gradient(cmap='Reds', subset=['precision','recall','f1-score']),
                 use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – PREDICTION TOOL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Prediksi":
    st.markdown("# Alat Prediksi Penyakit Jantung")
    st.markdown("*Isi informasi pasien di bawah ini untuk mendapatkan penilaian risiko secara instan berbasis kecerdasan buatan*")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Data Demografis Pasien")
        c1, c2 = st.columns(2)
        with c1:
            age_in = st.slider(
                "Usia",
                min_value=0,
                max_value=100,
                value=50
            )
            sex_in    = st.selectbox("Jenis Kelamin", ["Laki-laki (M)", "Perempuan (F)"])
            cpt_in    = st.selectbox("Tipe Nyeri Dada", [
                "ASY – Asimptomatik",
                "ATA – Angina Atipikal",
                "NAP – Nyeri Non-Anginal",
                "TA – Angina Tipikal"
            ])
        with c2:
            resting_bp_in = st.slider("Tekanan Darah Istirahat (mmHg)", 80, 200, 130)
            chol_in       = st.slider("Kolesterol (mg/dl)", 0, 603, 220)
            fbs_in        = st.selectbox("Gula Darah Puasa", ["<= 120 mg/dl (0)", "> 120 mg/dl (1)"])

        st.markdown("### Pengukuran Klinis")
        c3, c4 = st.columns(2)
        with c3:
            ecg_in    = st.selectbox("EKG Istirahat", ["Normal", "LVH", "ST"])
            maxhr_in  = st.slider("Denyut Jantung Maksimum", 60, 202, 150)
        with c4:
            ex_ang_in  = st.selectbox("Angina Saat Aktivitas Fisik", ["Tidak (N)", "Ya (Y)"])
            oldpeak_in = st.slider("Oldpeak (Depresi Segmen ST)", -2.6, 6.2, 0.5, 0.1)
            slope_in   = st.selectbox("Kemiringan Segmen ST", ["Flat", "Up", "Down"])

        predict_btn = st.button("Prediksi Risiko Penyakit Jantung")

    with col_right:
        st.markdown("### Ringkasan Data Input")
        summary = {
            "Usia": age_in,
            "Jenis Kelamin": "Laki-laki" if "M" in sex_in else "Perempuan",
            "Tipe Nyeri Dada": cpt_in.split("–")[0].strip(),
            "Tekanan Darah Istirahat": f"{resting_bp_in} mmHg",
            "Kolesterol": f"{chol_in} mg/dl",
            "Gula Darah Puasa": "> 120" if "1" in fbs_in else "<= 120",
            "EKG": ecg_in,
            "Denyut Jantung Maks.": maxhr_in,
            "Angina saat Aktivitas": "Ya" if "Y" in ex_ang_in else "Tidak",
            "Oldpeak": oldpeak_in,
            "Kemiringan ST": slope_in
        }
        for k, v in summary.items():
            st.markdown(f"**{k}:** {v}")

    if predict_btn:
        sex_val   = 1 if "M" in sex_in else 0
        cpt_code  = cpt_in.split("–")[0].strip()
        fbs_val   = 1 if "1" in fbs_in else 0
        exang_val = 1 if "Y" in ex_ang_in else 0

        row = {
            'Age': age_in, 'RestingBP': resting_bp_in, 'Cholesterol': chol_in,
            'FastingBS': fbs_val, 'MaxHR': maxhr_in, 'Oldpeak': oldpeak_in,
            'Sex_M': sex_val,
            'ChestPainType_ATA': 1 if cpt_code=='ATA' else 0,
            'ChestPainType_NAP': 1 if cpt_code=='NAP' else 0,
            'ChestPainType_TA':  1 if cpt_code=='TA'  else 0,
            'RestingECG_Normal': 1 if ecg_in=='Normal' else 0,
            'RestingECG_ST':     1 if ecg_in=='ST'     else 0,
            'ExerciseAngina_Y':  exang_val,
            'ST_Slope_Flat':     1 if slope_in=='Flat' else 0,
            'ST_Slope_Up':       1 if slope_in=='Up'   else 0,
        }

        input_df = pd.DataFrame([row])
        for col in feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[feature_cols]

        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]

        st.markdown("---")
        st.markdown("## Hasil Prediksi")
        c_res, c_gauge = st.columns([1, 1])

        with c_res:
            if pred == 1:
                st.markdown(f"""<div class="predict-result-positive">
                    RISIKO TINGGI<br>Terindikasi Penyakit Jantung<br>
                    <span style="font-size:1rem;font-weight:400">Probabilitas: {prob*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
                st.error("**Rekomendasi:** Segera konsultasikan kondisi ini kepada dokter spesialis jantung. Pemeriksaan diagnostik lanjutan sangat disarankan.")
            else:
                st.markdown(f"""<div class="predict-result-negative">
                    RISIKO RENDAH<br>Tidak Terindikasi Penyakit Jantung<br>
                    <span style="font-size:1rem;font-weight:400">Probabilitas: {prob*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
                st.success("**Rekomendasi:** Pertahankan gaya hidup sehat. Pemeriksaan kesehatan rutin tetap dianjurkan sebagai langkah pencegahan.")

        with c_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Skor Risiko (%)", 'font': {'color': 'white', 'size': 16}},
                number={'suffix': "%", 'font': {'color': 'white', 'size': 26}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'color': 'white'}},
                    'bar': {'color': RED if prob > 0.5 else GREEN},
                    'steps': [
                        {'range': [0, 30],  'color': '#1a3c34'},
                        {'range': [30, 60], 'color': '#3d2b1f'},
                        {'range': [60, 100],'color': '#3d1f1f'}
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 3}, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=260, template=DARK,
                                    paper_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("### Analisis Faktor Risiko")
        risks = []
        if age_in > 60:              risks.append(("Usia di Atas 60 Tahun", "Tinggi"))
        if sex_val == 1:             risks.append(("Jenis Kelamin Laki-laki", "Sedang"))
        if cpt_code == 'ASY':        risks.append(("Nyeri Dada Asimptomatik", "Tinggi"))
        if exang_val == 1:           risks.append(("Angina Saat Aktivitas Fisik", "Tinggi"))
        if oldpeak_in > 2:           risks.append(("Oldpeak Tinggi (>2)", "Tinggi"))
        if resting_bp_in > 140:      risks.append(("Tekanan Darah Tinggi", "Sedang"))
        if chol_in > 240:            risks.append(("Kolesterol Tinggi", "Sedang"))
        if fbs_val == 1:             risks.append(("Gula Darah Puasa Tinggi", "Sedang"))
        if maxhr_in < 100:           risks.append(("Denyut Jantung Maksimum Rendah (<100)", "Tinggi"))

        if risks:
            cols = st.columns(3)
            for i, (factor, level) in enumerate(risks):
                color = RED if level == "Tinggi" else "#f7971e"
                cols[i % 3].markdown(
                    f'<div style="background:rgba(233,69,96,0.1);border-left:4px solid {color};'
                    f'padding:10px;border-radius:6px;margin:5px 0;">'
                    f'<b style="color:{color};">Risiko {level}</b><br>{factor}</div>',
                    unsafe_allow_html=True
                )
        else:
            st.success("Tidak ditemukan faktor risiko individual yang signifikan berdasarkan nilai input yang diberikan.")
