import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dry Bean Explorer",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    :root {
        --bg-primary: #0e1117;
        --bg-card: #1a1d23;
        --accent: #4ade80;
        --accent-dim: #22c55e;
        --text-primary: #e2e8f0;
        --text-secondary: #94a3b8;
        --border: #2d333b;
    }

    .block-container { padding-top: 2rem; }

    .hero {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 50%, #022c22 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid #065f46;
    }
    .hero h1 {
        color: #ffffff !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.3rem !important;
    }
    .hero p {
        color: #a7f3d0 !important;
        font-size: 1.05rem !important;
        margin: 0 !important;
    }

    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card h3 {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem !important;
    }
    .metric-card .value {
        color: var(--accent) !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }

    .section-title {
        color: var(--text-primary) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin: 2rem 0 0.8rem 0 !important;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid var(--accent);
        display: inline-block;
    }

    div[data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem;
    }
    div[data-testid="stMetric"] label { color: var(--text-secondary) !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--accent) !important; }

    .stTabs [data-baseweb="tab-list"] { gap: 0; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.7rem 1.5rem;
        border-radius: 8px 8px 0 0;
    }

    div[data-testid="stSidebar"] {
        background: #111318;
        border-right: 1px solid var(--border);
    }

    .dataframe-container { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "raw" / "Dry_Bean.csv"
    df = pd.read_csv(csv_path)
    df["Class"] = df["Class"].astype(str).str.strip()
    return df


df = load_data()
classes = sorted(df["Class"].unique())
numeric_cols = [c for c in df.columns if c != "Class"]


# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🫘 Dry Bean Explorer</h1>
    <p>Clasificación Multiclase de Semillas de Judías — Análisis interactivo del dataset</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Filtros")
    selected_classes = st.multiselect(
        "Clases de judía",
        options=classes,
        default=classes,
        help="Selecciona una o más clases para filtrar el análisis",
    )
    st.divider()
    sample_pct = st.slider(
        "% de muestra para gráficos",
        min_value=10,
        max_value=100,
        value=100,
        step=10,
        help="Reduce para acelerar los gráficos con muchos puntos",
    )
    st.divider()
    st.markdown(
        f"**Muestras seleccionadas:** {len(df[df['Class'].isin(selected_classes)]):,} / {len(df):,}"
    )

# Filtered dataset
df_filtered = df[df["Class"].isin(selected_classes)].copy()
if sample_pct < 100:
    df_sample = df_filtered.sample(frac=sample_pct / 100, random_state=42)
else:
    df_sample = df_filtered


# ── KPI Cards ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Resumen General</p>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Muestras totales", f"{len(df_filtered):,}")
with col2:
    st.metric("Clases activas", len(selected_classes))
with col3:
    st.metric("Features numéricas", len(numeric_cols))
with col4:
    st.metric("Duplicados", f"{df_filtered.duplicated().sum():,}")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dist, tab_rel, tab_heat, tab_stats, tab_data = st.tabs(
    ["📊 Distribución", "🔍 Relaciones", "🌡️ Correlación", "📈 Estadísticas", "🗃️ Datos"]
)


# ── TAB 1: Distribution ───────────────────────────────────────────────────────
with tab_dist:
    st.markdown("### Distribución de Clases")

    fig_bar = px.bar(
        df_filtered["Class"].value_counts().reset_index(),
        x="Class",
        y="count",
        color="Class",
        color_discrete_sequence=px.colors.qualitative.Set2,
        text="count",
        labels={"count": "Frecuencia", "Class": "Tipo de Judía"},
    )
    fig_bar.update_traces(
        textposition="outside",
        textfont_size=13,
        marker_line_width=0,
    )
    fig_bar.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=False,
        height=420,
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### Distribución de Variables Numéricas")

    selected_feature = st.selectbox(
        "Selecciona una feature", numeric_cols, index=0, key="dist_feature"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_hist = px.histogram(
            df_sample,
            x=selected_feature,
            color="Class",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            opacity=0.75,
            barmode="overlay",
            labels={selected_feature: selected_feature, "count": "Frecuencia"},
        )
        fig_hist.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=380,
            margin=dict(t=30, b=30),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        fig_box = px.box(
            df_sample,
            x="Class",
            y=selected_feature,
            color="Class",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_box.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            height=380,
            margin=dict(t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ── TAB 2: Relationships ──────────────────────────────────────────────────────
with tab_rel:
    st.markdown("### Relaciones entre Features")

    col_x, col_y = st.columns(2)
    with col_x:
        feat_x = st.selectbox("Eje X", numeric_cols, index=0, key="feat_x")
    with col_y:
        feat_y = st.selectbox(
            "Eje Y", numeric_cols, index=6 if len(numeric_cols) > 6 else 0, key="feat_y"
        )

    fig_scatter = px.scatter(
        df_sample,
        x=feat_x,
        y=feat_y,
        color="Class",
        opacity=0.65,
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={feat_x: feat_x, feat_y: feat_y},
    )
    fig_scatter.update_traces(marker=dict(size=5, line=dict(width=0.3, color="#1a1d23")))
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=520,
        margin=dict(t=30, b=30),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("### Violin Plot — Comparación por Clase")

    violin_feat = st.selectbox(
        "Feature para violin plot", numeric_cols, index=0, key="violin_feat"
    )
    fig_violin = px.violin(
        df_sample,
        x="Class",
        y=violin_feat,
        color="Class",
        box=True,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_violin.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=420,
        margin=dict(t=30, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_violin, use_container_width=True)


# ── TAB 3: Correlation ────────────────────────────────────────────────────────
with tab_heat:
    st.markdown("### Matriz de Correlación")

    corr_method = st.radio(
        "Método", ["pearson", "spearman", "kendall"], horizontal=True
    )
    corr = df_filtered[numeric_cols].corr(method=corr_method)

    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
        text_auto=".2f",
    )
    fig_heat.update_layout(
        font_color="#e2e8f0",
        height=650,
        margin=dict(t=40, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("### Pares más correlacionados (excluyendo diagonal)")
    corr_pairs = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
    )
    corr_pairs.columns = ["Feature 1", "Feature 2", "Correlación"]
    corr_pairs["abs_corr"] = corr_pairs["Correlación"].abs()
    top_pairs = corr_pairs.sort_values("abs_corr", ascending=False).head(10)

    fig_pairs = px.bar(
        top_pairs,
        x="Correlación",
        y=top_pairs.apply(lambda r: f"{r['Feature 1']} ↔ {r['Feature 2']}", axis=1),
        orientation="h",
        color="Correlación",
        color_continuous_scale="RdBu_r",
        range_x=[-1, 1],
        labels={"y": "Par de Features", "Correlación": "Coef. de Correlación"},
    )
    fig_pairs.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        height=400,
        margin=dict(t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_pairs, use_container_width=True)


# ── TAB 4: Statistics ─────────────────────────────────────────────────────────
with tab_stats:
    st.markdown("### Estadísticas Descriptivas por Clase")

    stats = (
        df_filtered.groupby("Class")[numeric_cols]
        .agg(["mean", "std", "min", "median", "max"])
        .round(2)
    )

    stat_option = st.radio(
        "Métrica", ["mean", "std", "min", "median", "max"], horizontal=True
    )

    display_df = stats.xs(stat_option, axis=1, level=1).T
    display_df.index.name = "Feature"

    st.dataframe(
        display_df.style.background_gradient(
            cmap="Greens",
            axis=1,
        ).format(precision=2),
        use_container_width=True,
        height=500,
    )

    st.markdown("### Distribución de la Proporción entre Clases")
    class_counts = df_filtered["Class"].value_counts()
    fig_pie = px.pie(
        names=class_counts.index,
        values=class_counts.values,
        color=class_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig_pie.update_traces(textinfo="percent+label", textfont_size=12)
    fig_pie.update_layout(
        font_color="#e2e8f0",
        height=450,
        margin=dict(t=30, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ── TAB 5: Raw Data ───────────────────────────────────────────────────────────
with tab_data:
    st.markdown("### Explorador de Datos")

    col_search, col_sort = st.columns([2, 1])
    with col_search:
        search = st.text_input("🔍 Buscar en el dataset", placeholder="Ej: 20000")
    with col_sort:
        sort_col = st.selectbox("Ordenar por", df_filtered.columns, index=0)

    display_data = df_filtered.copy()
    if search:
        mask = display_data.apply(
            lambda row: row.astype(str).str.contains(search, case=False, na=False).any(),
            axis=1,
        )
        display_data = display_data[mask]

    display_data = display_data.sort_values(sort_col)

    st.markdown(f"**{len(display_data):,} registros**")
    st.dataframe(display_data, use_container_width=True, height=450)

    st.download_button(
        label="⬇️ Descargar datos filtrados (CSV)",
        data=display_data.to_csv(index=False).encode("utf-8"),
        file_name="dry_beans_filtered.csv",
        mime="text/csv",
    )
