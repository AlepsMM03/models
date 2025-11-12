"""
Dashboard avanzado para comparar optimizadores, activaciones y arquitecturas
de una MLP entrenada con PCA.

Ejecución:
    streamlit run dashboard.py
"""

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# =================== CONFIGURACIÓN ===================
st.set_page_config(
    page_title="Análisis de Optimizadores y Arquitecturas - MLP con PCA",
    layout="wide"
)

# Tema oscuro elegante
st.markdown("""
<style>
body { background-color: #0E1117; color: #E0E0E0; }
.stApp { background-color: #0E1117; }
h1, h2, h3, h4, h5, h6 { color: #E0E0E0; font-family: 'Inter', sans-serif; }
.stDataFrame { background-color: #1E2228 !important; }
</style>
""", unsafe_allow_html=True)


# =================== FUNCIÓN DE CARGA ===================
def load_results_local(folder_path="results"):
    """
    Carga los resultados desde archivos JSON en una carpeta local
    """
    try:
        # Verificar si la carpeta existe
        if not os.path.exists(folder_path):
            st.warning(f"La carpeta '{folder_path}' no existe. Creándola...")
            os.makedirs(folder_path)
            return pd.DataFrame()
        
        # Obtener todos los archivos JSON en la carpeta
        json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
        
        if not json_files:
            st.warning(f"No se encontraron archivos JSON en la carpeta '{folder_path}'.")
            return pd.DataFrame()
        
        data = []
        for file_name in json_files:
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    j = json.load(f)
                
                # Procesar los datos como antes
                if isinstance(j.get("arch"), list):
                    j["arch"] = str(j["arch"])
                for key in ["train_loss", "val_f1"]:
                    if key not in j or not isinstance(j[key], list):
                        j[key] = []
                
                data.append(j)
                
            except Exception as e:
                st.error(f"Error leyendo {file_name}: {e}")
        
        st.success(f"Se cargaron {len(data)} archivos JSON desde '{folder_path}'")
        return pd.DataFrame(data)
        
    except Exception as e:
        st.error(f"Error al acceder a la carpeta local: {e}")
        return pd.DataFrame()


# =================== INTERFAZ PRINCIPAL ===================
st.title("Comparación de Optimizadores, Activaciones y Arquitecturas en una MLP (con PCA)")
st.caption("Visualización interactiva basada en los experimentos realizados sobre el dataset de estados mentales.")

# Input para la ruta de la carpeta
default_folder = "results"  # Puedes cambiar esta ruta por defecto
folder_path = st.text_input("Ruta de la carpeta con los archivos JSON:", value=default_folder)

if folder_path:
    df = load_results_local(folder_path)
else:
    df = pd.DataFrame()

if df.empty:
    st.stop()

# =================== FILTROS ===================
optims = sorted(df["optimizer"].dropna().unique())
acts = sorted(df["activation"].dropna().unique())
archs = sorted(df["arch"].dropna().unique())

cols = st.columns(3)
optims_sel = cols[0].multiselect("Optimizadores:", optims, default=optims)
acts_sel = cols[1].multiselect("Activaciones:", acts, default=acts)
archs_sel = cols[2].multiselect("Arquitecturas:", archs, default=archs)

filtered = df[
    df["optimizer"].isin(optims_sel)
    & df["activation"].isin(acts_sel)
    & df["arch"].isin(archs_sel)
].copy()

if filtered.empty:
    st.warning("No hay resultados que coincidan con los filtros seleccionados.")
    st.stop()

# =================== TABLA RESUMEN ===================
st.subheader("Resultados Globales del Conjunto de Prueba")
st.dataframe(
    filtered[["optimizer", "activation", "arch", "test_acc", "test_f1", "runtime"]]
    .sort_values("test_f1", ascending=False)
    .reset_index(drop=True),
    use_container_width=True
)

# =================== VISUALIZACIONES ===================
st.markdown("---")
st.subheader("Comparaciones entre Combinaciones")

col1, col2 = st.columns(2)

with col1:
    fig_f1 = px.bar(
        filtered,
        x="optimizer",
        y="test_f1",
        color="activation",
        facet_col="arch",
        title="F1-score por combinación",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_f1.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
    st.plotly_chart(fig_f1, use_container_width=True)

with col2:
    fig_acc = px.bar(
        filtered,
        x="optimizer",
        y="test_acc",
        color="activation",
        facet_col="arch",
        title="Accuracy por combinación",
        text_auto=".3f",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_acc.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
    st.plotly_chart(fig_acc, use_container_width=True)

# =================== DISPERSIÓN F1 VS ACC ===================
st.markdown("---")
st.subheader("Relación entre F1-score y Accuracy")

fig_scatter = px.scatter(
    filtered,
    x="test_acc",
    y="test_f1",
    color="optimizer",
    symbol="activation",
    size="runtime",
    hover_data=["arch"],
    title="Dispersión de desempeño entre Accuracy y F1",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_scatter.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
st.plotly_chart(fig_scatter, use_container_width=True)

# =================== HEATMAP ===================
st.markdown("---")
st.subheader("Mapa de calor: F1 promedio por Optimizador y Activación")

heatmap_data = (
    filtered.groupby(["optimizer", "activation"])["test_f1"]
    .mean()
    .reset_index()
    .pivot(index="activation", columns="optimizer", values="test_f1")
)

fig_heatmap = px.imshow(
    heatmap_data,
    text_auto=".2f",
    color_continuous_scale="viridis",
    title="F1-score promedio por combinación"
)
fig_heatmap.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
st.plotly_chart(fig_heatmap, use_container_width=True)

# =================== CURVAS DE ENTRENAMIENTO ===================
st.markdown("---")
st.subheader("Curvas de Entrenamiento Individuales")

filtered["combo"] = filtered.apply(lambda r: f"{r['optimizer']} | {r['activation']} | {r['arch']}", axis=1)
selected = st.selectbox("Selecciona una combinación:", filtered["combo"])

row = filtered.loc[filtered["combo"] == selected].iloc[0]
if row["train_loss"] and row["val_f1"]:
    epochs = list(range(1, len(row["train_loss"]) + 1))
    df_curve = pd.DataFrame({
        "Época": epochs,
        "Pérdida (train)": row["train_loss"],
        "F1 (val)": row["val_f1"]
    })
    fig_curve = px.line(
        df_curve,
        x="Época",
        y=["Pérdida (train)", "F1 (val)"],
        markers=True,
        title=f"Evolución del entrenamiento: {selected}",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_curve.update_layout(template="plotly_dark", plot_bgcolor="#0E1117", paper_bgcolor="#0E1117")
    st.plotly_chart(fig_curve, use_container_width=True)
else:
    st.warning("No se encontraron curvas de entrenamiento para esta combinación.")


# =================== VISUALIZACIONES 3D ===================
st.markdown("---")
st.subheader("Visualizaciones 3D")

# --- 1. Nube 3D de rendimiento por combinación ---
st.markdown("**Nube 3D de rendimiento (Accuracy, F1, Runtime)**")

fig_3d_perf = px.scatter_3d(
    filtered,
    x="test_acc",
    y="test_f1",
    z="runtime",
    color="optimizer",
    size="pca_components",
    symbol="activation",
    hover_name="arch",
    title="Espacio de rendimiento de modelos (3D)",
    color_discrete_sequence=px.colors.qualitative.Vivid
)
fig_3d_perf.update_layout(
    template="plotly_dark",
    scene=dict(
        xaxis_title="Accuracy",
        yaxis_title="F1-score",
        zaxis_title="Tiempo de ejecución (s)",
        bgcolor="#0E1117",
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)
st.plotly_chart(fig_3d_perf, use_container_width=True)


# ==========================================
# DESCENSO DE GRADIENTE (Evolución del Entrenamiento) — INTERACTIVO
# ==========================================
import plotly.graph_objects as go

st.markdown("---")
st.subheader("📉 Descenso de Gradiente (Evolución del Entrenamiento)")

# Seleccionar el experimento
combo_exp = st.selectbox(
    "Selecciona el experimento para visualizar el descenso de gradiente:",
    filtered["combo"],
    key="descenso_gradiente"
)

# Obtener los datos del experimento seleccionado
exp_data = filtered.loc[filtered["combo"] == combo_exp].iloc[0]
train_losses = exp_data["train_loss"]
val_f1 = exp_data["val_f1"]

if isinstance(train_losses, list) and len(train_losses) > 0:
    epochs = list(range(1, len(train_losses) + 1))

    # Figura interactiva
    fig = go.Figure()

    # Curva de pérdida (Loss)
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses,
        mode='lines+markers',
        name='Pérdida (Entrenamiento)',
        line=dict(color='#4A90E2', width=3),
        marker=dict(size=6, color='#4A90E2', symbol='circle')
    ))

    # Curva de F1 (Validación)
    if isinstance(val_f1, list) and len(val_f1) > 0:
        val_f1_scaled = [f * max(train_losses) for f in val_f1]  # Escalar F1 visualmente
        fig.add_trace(go.Scatter(
            x=epochs, y=val_f1_scaled,
            mode='lines+markers',
            name='F1 Validación (Escalada)',
            line=dict(color='#7ED321', width=2, dash='dash'),
            marker=dict(size=6, color='#7ED321', symbol='diamond')
        ))

    # Personalización visual avanzada
    fig.update_layout(
        title=f"Descenso de Gradiente — {exp_data['optimizer']} | {exp_data['activation']} | {exp_data['arch']}",
        xaxis_title="Épocas",
        yaxis_title="Magnitud",
        template="plotly_dark",
        hovermode="x unified",
        font=dict(size=13),
        height=500,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.2)"
        ),
        margin=dict(l=40, r=40, b=40, t=60)
    )

    # Animación interactiva opcional (puntos que avanzan por época)
    frames = [go.Frame(data=[
        go.Scatter(
            x=epochs[:k],
            y=train_losses[:k],
            mode="lines+markers",
            line=dict(color='#4A90E2', width=3),
            marker=dict(size=6, color='#4A90E2')
        ),
        go.Scatter(
            x=epochs[:k],
            y=val_f1_scaled[:k] if len(val_f1) == len(epochs) else [],
            mode="lines+markers",
            line=dict(color='#7ED321', width=2, dash='dash'),
            marker=dict(size=6, color='#7ED321')
        )
    ], name=str(k)) for k in range(1, len(epochs)+1)]

    fig.update(frames=frames)
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {"args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}],
                 "label": "▶️ Reproducir", "method": "animate"},
                {"args": [[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}],
                 "label": "⏸️ Pausar", "method": "animate"}
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 40},
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 1.2,
            "yanchor": "top"
        }]
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No hay datos suficientes de entrenamiento para graficar el descenso de gradiente.")


# ==========================================
# DESCENSO DE GRADIENTE 3D INTERACTIVO
# ==========================================
st.markdown("---")
st.subheader("🌌 Visualización 3D del Descenso de Gradiente")

# Selección de combinación
combo_sel = st.selectbox("Selecciona el experimento para visualizar:", filtered["combo"])

exp_row = filtered.loc[filtered["combo"] == combo_sel].iloc[0]

train_loss = exp_row["train_loss"]
val_f1 = exp_row["val_f1"]

if len(train_loss) > 1 and len(val_f1) > 1:
    epochs = list(range(1, len(train_loss) + 1))

    # Crear la superficie 3D interpolada del descenso de gradiente
    X, Y = np.meshgrid(epochs, np.linspace(0, 1, len(epochs)))
    Z = np.array(train_loss)
    Z = np.tile(Z, (len(epochs), 1))

    fig_sdg = go.Figure()

    # Superficie suave del descenso de pérdida
    fig_sdg.add_trace(go.Surface(
        z=Z, x=X, y=Y,
        colorscale="Viridis",
        opacity=0.8,
        name="Superficie de pérdida",
        showscale=True
    ))

    # Línea del F1 sobre la superficie (trayectoria de validación)
    fig_sdg.add_trace(go.Scatter3d(
        x=epochs,
        y=[0.5]*len(val_f1),
        z=val_f1,
        mode="lines+markers",
        line=dict(color="cyan", width=5),
        marker=dict(size=4, color="white"),
        name="F1 (Validación)"
    ))

    fig_sdg.update_layout(
        title=f"Descenso de Gradiente 3D — {exp_row['optimizer']} | {exp_row['activation']} | {exp_row['arch']}",
        scene=dict(
            xaxis_title="Épocas",
            yaxis_title="Eje auxiliar",
            zaxis_title="Valor",
            bgcolor="#0E1117",
            xaxis=dict(gridcolor="#222", showbackground=True, backgroundcolor="#111"),
            yaxis=dict(gridcolor="#222", showbackground=True, backgroundcolor="#111"),
            zaxis=dict(gridcolor="#222", showbackground=True, backgroundcolor="#111"),
        ),
        template="plotly_dark",
        height=700,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    st.plotly_chart(fig_sdg, use_container_width=True)
else:
    st.warning("Este experimento no tiene suficientes datos para graficar el descenso de gradiente 3D.")


# =================== INTERPRETACIÓN FINAL ===================
st.markdown("---")
st.subheader("Interpretación Automática")

best = filtered.loc[filtered["test_f1"].idxmax()]
st.info(
    f"El mejor modelo fue entrenado con {best['optimizer']}, activación {best['activation']} "
    f"y arquitectura {best['arch']}, alcanzando un F1 = {best['test_f1']:.3f} "
    f"y Accuracy = {best['test_acc']:.3f}."
)
