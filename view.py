#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
kepler_grokking_monitor.py

Autor: Gris Iscomeback
Descripción: Visualizador interactivo de grokking para órbitas keplerianas usando app.py
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import fft
from datetime import datetime
import time
import os
import sys

# Aseguramos que el directorio actual esté en el path para importar app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ================================
# IMPORTACIÓN DIRECTA DE TU CÓDIGO
# ================================
try:
    from app import (
        generate_kepler_orbits,
        KeplerOrbitPredictor,
        train_until_grok,
        expand_model_weights_geometric,
        analyze_geometric_representation
    )
except ImportError as e:
    st.error(f"❌ Error al importar funciones desde 'app.py':\n{e}")
    st.stop()

# Desactivar advertencias gráficas si no hay display
if 'DISPLAY' not in os.environ and os.name != 'nt':
    import matplotlib
    matplotlib.use('Agg')

# ================================
# CONFIGURACIÓN DE STREAMLIT
# ================================
st.set_page_config(
    page_title="🌌 Kepler Grokking Monitor",
    layout="wide",
    page_icon="🪐",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0a0e17 0%, #1a2330 100%);
        color: #e0e0ff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header-container {
        background: rgba(15, 32, 61, 0.85);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #4a6fa5;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .metric-card {
        background: rgba(26, 42, 85, 0.7);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #3a5ba0;
        transition: all 0.3s ease;
    }
    .phase-indicator {
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 0.5rem 0;
        background: rgba(30, 60, 120, 0.6);
        border: 1px solid #5c9bd5;
    }
    .scientific-notation {
        font-family: 'Lucida Console', Monaco, monospace;
        background: rgba(30, 45, 80, 0.6);
        padding: 0.5rem;
        border-radius: 8px;
        border-left: 3px solid #4a86e8;
        margin: 0.5rem 0;
    }
    .theory-box {
        background: rgba(19, 41, 77, 0.85);
        border: 1px solid #5c9bd5;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    h1, h2, h3 {
        color: #64b5f6 !important;
        text-shadow: 0 0 8px rgba(100, 181, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ================================
# ESTADO DE SESIÓN
# ================================
if 'training_done' not in st.session_state:
    st.session_state.training_done = False
    st.session_state.base_model = None
    st.session_state.expanded_model = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.X_large_test = None
    st.session_state.y_large_test = None
    st.session_state.history = None
    st.session_state.grok_achieved = False

# ================================
# FUNCIÓN DE MÉTRICAS GEOMÉTRICAS
# ================================
def calculate_kepler_metrics(model, X_sample):
    """Calcula métricas basadas en la representación interna del modelo real"""
    result = analyze_geometric_representation(model, X_sample)
    return {
        'angle_consistency': result['angle_consistency_h2'],
        'distance_preservation': result['distance_preservation'],
        'structural_stability': result['representation_stability'],
        'fractal_dim': min(50, int(result['distance_preservation'] * 40) + 3)
    }

# ================================
# INTERFAZ PRINCIPAL
# ================================
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.title("🪐 Kepler Grokking Dynamics Monitor")
st.subheader("Visualización Científica del Aprendizaje Algorítmico en Órbitas Keplerianas")
st.markdown("""
<div class="theory-box">
<b>Base Física:</b> Este monitor entrena redes neuronales para predecir la evolución de órbitas keplerianas. 
Cuando se logra <i>grokking</i>, la red no memoriza trayectorias, sino que descubre la <b>estructura geométrica profunda</b> 
(conservación del momento angular, simetría rotacional) que subyace a la dinámica orbital.
</div>
</div>
""", unsafe_allow_html=True)

# ================================
# CONTROLES EN SIDEBAR
# ================================
with st.sidebar:
    st.header("⚙️ Configuración del Experimento")
    st.markdown("---")
    
    hidden_base = st.slider("Unidades ocultas (base)", 64, 256, 128, step=64)
    scale_factor = st.slider("Factor de expansión", 1, 4, 2)
    noise_level = st.number_input("Nivel de ruido", 0.0001, 0.01, 0.0005, format="%.4f")
    grok_threshold = st.number_input("Umbral de grokking", 1e-6, 1e-3, 5e-5, format="%.1e")
    
    if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
        with st.spinner("Generando datos y entrenando modelo..."):
            # 1. Generar datos
            X, y = generate_kepler_orbits(n_samples=2000, noise_level=noise_level, max_time=3.0, seed=42)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 2. Entrenar modelo base
            base_model = KeplerOrbitPredictor(input_size=5, hidden_size=hidden_base, output_size=2)
            model, history, grok_achieved = train_until_grok(
                base_model, X_train, y_train, X_test, y_test,
                max_epochs=15000,
                patience=1500,
                initial_lr=0.003,
                weight_decay=5e-6,
                grok_threshold=grok_threshold
            )
            
            # 3. Expansión geométrica
            expanded_model = expand_model_weights_geometric(model, scale_factor=scale_factor)
            
            # 4. Datos escalados para transferencia
            X_large, y_large = generate_kepler_orbits(n_samples=1000, noise_level=noise_level*4, max_time=8.0, seed=123)
            _, X_large_test, _, y_large_test = train_test_split(X_large, y_large, test_size=0.5, random_state=42)
            
            # 5. Guardar en sesión
            st.session_state.base_model = model
            st.session_state.expanded_model = expanded_model
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_large_test = X_large_test
            st.session_state.y_large_test = y_large_test
            st.session_state.history = history
            st.session_state.grok_achieved = grok_achieved
            st.session_state.training_done = True
        
        st.success("✅ Entrenamiento completado")

# ================================
# VISUALIZACIONES DINÁMICAS
# ================================
if st.session_state.training_done:
    base_model = st.session_state.base_model
    expanded_model = st.session_state.expanded_model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    X_large_test = st.session_state.X_large_test
    y_large_test = st.session_state.X_large_test  # solo para métricas, no predicción
    
    # Evaluar modelos
    with torch.no_grad():
        base_pred = base_model(torch.FloatTensor(X_test)).numpy()
        expanded_pred = expanded_model(torch.FloatTensor(X_test)).numpy()
        base_mse = np.mean((base_pred - y_test)**2)
        expanded_mse = np.mean((expanded_pred - y_test)**2)
    
    # Métricas geométricas
    base_metrics = calculate_kepler_metrics(base_model, X_test[:100])
    expanded_metrics = calculate_kepler_metrics(expanded_model, X_test[:100])
    
    # === FASE ACTUAL ===
    if st.session_state.grok_achieved:
        phase = "Grokking"
        phase_desc = "✅ Solución algorítmica descubierta: generalización perfecta"
    elif min(st.session_state.history['test_loss']) < grok_threshold * 10:
        phase = "Transición"
        phase_desc = "🔄 Estructura geométrica emergente"
    else:
        phase = "Memorización"
        phase_desc = "🧠 Ajuste a datos con ruido"
    
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.subheader(f"🌌 Estado del Sistema: {phase}")
    st.markdown(f"<div class='phase-indicator'>{phase_desc}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # PCA 3D de pesos de la primera capa
        W = base_model.net[0].weight.detach().cpu().numpy()
        pca = PCA(n_components=3)
        W_pca = pca.fit_transform(W)
        norms = np.linalg.norm(W, axis=1)
        
        fig = go.Figure(data=go.Scatter3d(
            x=W_pca[:,0], y=W_pca[:,1], z=W_pca[:,2],
            mode='markers',
            marker=dict(
                size=5,
                color=norms,
                colorscale='Plasma' if phase=="Grokking" else 'Viridis',
                colorbar=dict(title="Norma", orientation="h"),
                opacity=0.8
            )
        ))
        fig.update_layout(
            template="plotly_dark",
            height=500,
            title="Geometría de Pesos (Capa 1)",
            scene=dict(
                xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3',
                aspectmode='cube'
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📊 Métricas")
        st.metric("MSE Base", f"{base_mse:.2e}")
        st.metric("MSE Expandido", f"{expanded_mse:.2e}")
        st.metric("Ángulos (H2)", f"{base_metrics['angle_consistency']:.3f}")
        st.metric("Dim. Fractal", f"{base_metrics['fractal_dim']:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # === CURVA DE APRENDIZAJE ===
    history = st.session_state.history
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(y=history['train_loss'], mode='lines', name='Train', line=dict(color='cyan')))
    fig_loss.add_trace(go.Scatter(y=history['test_loss'], mode='lines', name='Test', line=dict(color='magenta')))
    fig_loss.update_layout(
        template="plotly_dark",
        title="Curva de Aprendizaje",
        xaxis_title="Época",
        yaxis_title="MSE (escala log)",
        yaxis_type="log"
    )
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # === ANÁLISIS ESPECTRAL ===
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.subheader("🔬 Análisis Espectral de la Solución")
    st.markdown('<div class="theory-box">La estructura algorítmica se manifiesta en frecuencias específicas en los pesos.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    neuron_idx = 0
    if neuron_idx < W.shape[0]:
        weights = W[neuron_idx]
        fft_vals = np.abs(fft.rfft(weights))
        freqs = fft.rfftfreq(len(weights))
        
        fig_fft = go.Figure()
        fig_fft.add_trace(go.Scatter(x=freqs[1:], y=fft_vals[1:], mode='lines', fill='tozeroy'))
        fig_fft.update_layout(
            template="plotly_dark",
            title=f"Espectro de Neurona {neuron_idx}",
            xaxis_title="Frecuencia",
            yaxis_title="Magnitud"
        )
        st.plotly_chart(fig_fft, use_container_width=True)
    
    # === INTERPRETACIÓN ===
    interpretation = {
        "Memorización": "La red ajusta pesos para reproducir trayectorias ruidosas sin capturar la invariancia física.",
        "Transición": "Empiezan a emerger direcciones en el espacio de pesos que codifican simetrías rotacionales y conservación.",
        "Grokking": "La representación interna colapsa a una solución geométrica mínima que respeta las invariantes de Kepler."
    }
    st.markdown(f"<div class='scientific-notation'>{interpretation[phase]}</div>", unsafe_allow_html=True)

else:
    st.info("👆 Usa el panel lateral para configurar y lanzar el entrenamiento.")
    st.markdown("""
    <div class="scientific-notation">
    Este monitor entrena una red para predecir órbitas keplerianas. Al alcanzar <b>grokking</b>, 
    la red descubre la <i>estructura simétrica subyacente</i> en lugar de memorizar datos.
    </div>
    """, unsafe_allow_html=True)

# ================================
# FOOTER
# ================================
st.markdown("""
<div class="footer" style="text-align:center; padding:1.5rem 0; color:#8a9bbd; border-top:1px solid #2c4a7d; margin-top:2rem;">
    <p>🌌 <b>Kepler Grokking Monitor</b> – Descubrimiento de Invariantes Físicas mediante Aprendizaje Algorítmico</p>
    <p>Basado en investigación de <i>grokking</i> y transferencia geométrica. Datos y modelo definidos en <code>app.py</code>.</p>
    <p>Generado: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
</div>
""", unsafe_allow_html=True)
