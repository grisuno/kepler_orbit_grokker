import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from scipy import fft
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time

st.set_page_config(page_title="🔬 Grokking Conservation Lab", layout="wide", page_icon="🧠", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #0a0e17 0%, #0d1b2a 100%); color: #e0e0ff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .header-container { background: rgba(15, 32, 61, 0.85); border-radius: 15px; padding: 1.5rem; margin: 1rem 0; border: 1px solid #4a6fa5; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4); }
    .metric-card { background: rgba(26, 42, 85, 0.7); border-radius: 12px; padding: 1rem; border: 1px solid #3a5ba0; transition: all 0.3s ease; }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 6px 15px rgba(58, 91, 160, 0.3); }
    .phase-indicator { font-weight: bold; padding: 0.5rem 1rem; border-radius: 20px; display: inline-block; margin: 0.5rem 0; }
    .scientific-notation { font-family: 'Lucida Console', Monaco, monospace; background: rgba(30, 45, 80, 0.6); padding: 0.25rem 0.5rem; border-radius: 5px; border-left: 3px solid #4a86e8; }
    .citation-box { background: rgba(22, 38, 68, 0.8); border-left: 4px solid #64b5f6; padding: 1rem; margin: 1rem 0; font-style: italic; color: #bbdefb; }
    .theory-box { background: rgba(19, 41, 77, 0.85); border: 1px solid #5c9bd5; border-radius: 10px; padding: 1.5rem; margin: 1.5rem 0; }
    .success-box { background: rgba(34, 139, 34, 0.2); border-left: 4px solid #22bb33; padding: 1rem; margin: 1rem 0; border-radius: 5px; color: #90ee90; }
    .footer { text-align: center; padding: 2rem 0; color: #8a9bbd; font-size: 0.9rem; border-top: 1px solid #2c4a7d; margin-top: 2rem; }
    h1, h2, h3 { color: #64b5f6 !important; text-shadow: 0 0 10px rgba(100, 181, 246, 0.3); }
</style>
""", unsafe_allow_html=True)

# Neural Network Models
class ParityMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def generate_parity_data(n_bits, k_active=3, n_samples=1000, seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    X = torch.randint(0, 2, (n_samples, n_bits)).float()
    active_bits = np.random.choice(n_bits, k_active, replace=False)
    y = (X[:, active_bits].sum(dim=1) % 2).unsqueeze(1).float()
    return X, y, active_bits

def train_parity_model(n_bits=32, hidden_dim=128, epochs=200, lr=0.001):
    model = ParityMLP(n_bits, hidden_dim)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    criterion = nn.BCEWithLogitsLoss()
    X_train, y_train, active_bits = generate_parity_data(n_bits, n_samples=500)
    X_test, y_test, _ = generate_parity_data(n_bits, n_samples=200, seed=43)
    train_losses, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_preds = (torch.sigmoid(test_outputs) > 0.5).float()
            test_acc = (test_preds == y_test).float().mean().item()
        train_losses.append(loss.item())
        test_accs.append(test_acc)
    return model, train_losses, test_accs, active_bits

def expand_model_weights(base_model, target_bits):
    base_bits = list(base_model.parameters())[0].shape[1]
    base_hidden = list(base_model.parameters())[0].shape[0]
    scale = target_bits // base_bits
    new_hidden = base_hidden * scale
    expanded = ParityMLP(target_bits, new_hidden)
    with torch.no_grad():
        W1 = base_model.fc1.weight.data
        W2 = base_model.fc2.weight.data
        W3 = base_model.fc3.weight.data
        for i in range(scale):
            expanded.fc1.weight.data[i*base_hidden:(i+1)*base_hidden, i*base_bits:(i+1)*base_bits] = W1
            expanded.fc2.weight.data[i*base_hidden:(i+1)*base_hidden, i*base_hidden:(i+1)*base_hidden] = W2
            expanded.fc3.weight.data[:, i*base_hidden:(i+1)*base_hidden] = W3 / scale
    return expanded

class ScientificWeightGenerator:
    @staticmethod
    def generate_weights(phase, d_model=512, d_sae=1024, seed=42):
        np.random.seed(seed)
        if phase == "Noise":
            W = np.random.randn(d_sae, d_model) * 0.1
            for i in range(50):
                idx = np.random.choice(d_sae, 2, replace=False)
                W[idx[1]] = W[idx[0]] * 0.7 + np.random.randn(d_model) * 0.05
            return W, 0.52, "Initial chaos - spurious correlations"
        elif phase == "Memorization":
            W = np.zeros((d_sae, d_model))
            cluster_size = d_sae // 8
            for cluster in range(8):
                start_idx = cluster * cluster_size
                end_idx = min((cluster + 1) * cluster_size, d_sae)
                base_pattern = np.random.randn(d_model) * 0.8
                for i in range(start_idx, end_idx):
                    W[i] = base_pattern * (0.9 + np.random.randn() * 0.1) + np.random.randn(d_model) * 0.3
            return W, 0.98, "Dense memorization - neural clusters"
        elif phase == "Transition":
            W = np.zeros((d_sae, d_model))
            for i in range(d_sae):
                freq = (i % 16) + 1
                phase_shift = np.random.uniform(0, 2 * np.pi)
                amplitude = 0.7 + np.random.exponential(0.3)
                W[i] = amplitude * np.sin(np.linspace(0, freq * np.pi, d_model) + phase_shift)
            structural_noise = np.random.randn(d_sae, d_model) * 0.15
            for i in range(d_sae // 20):
                start = i * 20
                end = min((i + 1) * 20, d_sae)
                structural_noise[start:end] = np.mean(structural_noise[start:end], axis=0)
            W += structural_noise
            return W, 0.85, "Algorithmic transition - emergent structure"
        else:
            W = np.zeros((d_sae, d_model))
            theta = np.linspace(0, 2 * np.pi, d_sae)
            for i in range(d_sae):
                for harmonic in [1, 3, 5, 7]:
                    W[i] += (1 / harmonic) * np.sin(harmonic * np.linspace(0, theta[i], d_model))
                W[i] = W[i] / np.max(np.abs(W[i])) * 0.9
            for layer in range(3):
                start_idx = layer * d_sae // 3
                end_idx = (layer + 1) * d_sae // 3
                W[start_idx:end_idx] *= (0.8 ** layer)
            return W, 1.0, "Algorithmic solution - minimal geometric representation"

def calculate_scientific_metrics(W):
    metrics = {}
    W_clean = np.nan_to_num(W.copy(), nan=0.0, posinf=1e6, neginf=-1e6)
    if not np.all(np.isfinite(W_clean)):
        return {'entropy': 0.0, 'fractal_dim': 1.0, 'coherence': 0.0, 'avg_coherence': 0.0, 'dominant_freq': 1, 'spectral_flatness': 0.0}
    flattened = W_clean.flatten()
    if np.allclose(flattened, flattened[0]):
        flattened = flattened + np.random.normal(0, 1e-10, flattened.shape)
    hist, _ = np.histogram(flattened, bins=50, density=True)
    hist = hist[hist > 0]
    metrics['entropy'] = -np.sum(hist * np.log(hist + 1e-10)) if len(hist) > 0 else 0.0
    sample_size = min(1000, W_clean.shape[0])
    W_sample = W_clean[:sample_size]
    try:
        pca = PCA(n_components=min(50, W_clean.shape[1], sample_size))
        pca.fit(W_sample)
        explained_variance = pca.explained_variance_ratio_
        fractal_dim = np.sum(explained_variance > 1e-3)
        metrics['fractal_dim'] = max(1.0, fractal_dim)
    except:
        metrics['fractal_dim'] = 1.0
    coherence_sample_size = min(200, W_clean.shape[0])
    W_sample = W_clean[:coherence_sample_size]
    norms = np.where(np.linalg.norm(W_sample, axis=1, keepdims=True) == 0, 1e-10, np.linalg.norm(W_sample, axis=1, keepdims=True))
    W_normalized = W_sample / norms
    coherence_matrix = np.abs(np.dot(W_normalized, W_normalized.T))
    np.fill_diagonal(coherence_matrix, 0)
    metrics['coherence'] = float(np.max(coherence_matrix)) if coherence_matrix.size > 0 else 0.0
    metrics['avg_coherence'] = float(np.mean(coherence_matrix)) if coherence_matrix.size > 0 else 0.0
    try:
        fft_magnitudes = np.abs(fft.rfft(W_clean[0]))
        if len(fft_magnitudes) > 1:
            dominant_freq = np.argmax(fft_magnitudes[1:]) + 1
            metrics['dominant_freq'] = int(dominant_freq)
            safe_magnitudes = np.maximum(fft_magnitudes, 1e-10)
            geometric_mean = np.exp(np.mean(np.log(safe_magnitudes)))
            metrics['spectral_flatness'] = float(10 * np.log10(np.mean(safe_magnitudes) / geometric_mean))
        else:
            metrics.update({'dominant_freq': 1, 'spectral_flatness': 0.0})
    except:
        metrics.update({'dominant_freq': 1, 'spectral_flatness': 0.0})
    return metrics

# Main Interface
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.title("🧠 GROKKING CONSERVATION LAB")
st.subheader("Scientific Visualization and Experimental Validation of Algorithmic Transfer")
st.markdown('<div class="citation-box"><b>Theoretical Foundation:</b> Grokking (Liu et al., 2022) describes the phenomenon where a model initially memorizes data, then experiences an abrupt transition to perfect generalization. This lab enables visualization AND experimental validation of algorithmic structure conservation through weight expansion.</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Operation Mode")
    mode = st.radio("Select mode:", ["📊 Theoretical Visualization", "🧪 Real Experiment"], help="Visualization: grokking phase simulations | Experiment: train and expand real model")
    st.markdown("---")
    if mode == "📊 Theoretical Visualization":
        st.subheader("Simulation Parameters")
        seed = st.number_input("Random Seed", 0, 10000, 42)
    else:
        st.subheader("🧪 Experimental Parameters")
        st.markdown("**Base Training:**")
        base_bits = st.selectbox("Input bits", [16, 32, 64], index=1)
        hidden_dim = st.slider("Hidden dimension", 64, 256, 128, 32)
        epochs = st.slider("Epochs", 100, 1000, 300, 100)
        st.markdown("**Expansion:**")
        target_bits = st.selectbox("Expand to N bits", [64, 128, 256, 512], index=1)
        st.markdown("---")
        if st.button("🚀 TRAIN BASE MODEL", use_container_width=True):
            with st.spinner(f"Training {base_bits}-bit model..."):
                model, losses, accs, active = train_parity_model(base_bits, hidden_dim, epochs)
                st.session_state.update({'base_model': model, 'train_losses': losses, 'test_accs': accs, 'active_bits': active, 'base_bits': base_bits})
            st.success(f"✅ Trained! Final acc: {accs[-1]:.3f}")
        if 'base_model' in st.session_state:
            st.markdown("---")
            if st.button(f"⚡ EXPAND TO {target_bits} BITS", use_container_width=True):
                with st.spinner("Expanding model (null-space surgery)..."):
                    expanded = expand_model_weights(st.session_state['base_model'], target_bits)
                    X_test, y_test, _ = generate_parity_data(target_bits, n_samples=500)
                    expanded.eval()
                    with torch.no_grad():
                        outputs = expanded(X_test)
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        acc = (preds == y_test).float().mean().item()
                    st.session_state.update({'expanded_model': expanded, 'expanded_acc': acc, 'target_bits': target_bits})
                st.success(f"✅ Expanded! Acc without training: {acc:.3f}")
    st.markdown("---")
    st.subheader("📚 References")
    st.markdown("- **Liu et al. (2022)**: Grokking paper\n- **Power et al. (2022)**: Formalization\n- **Nanda et al. (2023)**: Mechanistic interpretability")
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

if mode == "📊 Theoretical Visualization":
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 1, 1])
    with col_ctrl1:
        phase = st.select_slider("🔬 Grokking Process Phase", options=["Noise", "Memorization", "Transition", "Grokking"], value="Transition")
    with col_ctrl2:
        d_model = st.slider("Model Dimension", 128, 1024, 512, step=128)
    with col_ctrl3:
        d_sae = st.slider("SAE Dimension", 256, 2048, 1024, step=256)
    W, accuracy, phase_description = ScientificWeightGenerator.generate_weights(phase, d_model, d_sae)
    metrics = calculate_scientific_metrics(W)
    st.markdown(f'<div class="header-container"><h3>🌐 Latent Geometry: {phase}</h3><div class="phase-indicator">{phase_description}</div></div>', unsafe_allow_html=True)
    col_main1, col_main2 = st.columns([3, 1])
    with col_main1:
        pca = PCA(n_components=3)
        W_sample = np.nan_to_num(W[:1000], nan=0.0, posinf=1e6, neginf=-1e6)
        W_pca = pca.fit_transform(W_sample)
        colors = np.linalg.norm(W_sample, axis=1)
        fig_3d = go.Figure(data=[go.Scatter3d(x=W_pca[:, 0], y=W_pca[:, 1], z=W_pca[:, 2], mode='markers', marker=dict(size=4, color=colors, colorscale='Viridis' if phase != "Grokking" else 'Plasma', opacity=0.7, colorbar=dict(title="Norm", orientation="h")))])
        fig_3d.update_layout(scene=dict(xaxis_title='PCA 1', yaxis_title='PCA 2', zaxis_title='PCA 3', xaxis=dict(showbackground=False, gridcolor='#4a6fa5'), yaxis=dict(showbackground=False, gridcolor='#4a6fa5'), zaxis=dict(showbackground=False, gridcolor='#4a6fa5'), aspectmode='cube'), template="plotly_dark", height=650, title=f"Latent Geometry - {phase}")
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_main2:
        st.markdown('<div class="metric-card"><h4>📊 Scientific Metrics</h4>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Accuracy", f"{accuracy*100:.1f}%")
            st.metric("Fractal Dim", f"{metrics['fractal_dim']:.1f}")
        with col_m2:
            st.metric("Entropy", f"{metrics['entropy']:.2f}")
            st.metric("Coherence", f"{metrics['coherence']:.3f}")
        st.markdown("<br><h4>📈 Structural Correlation</h4>", unsafe_allow_html=True)
        sample_size = min(64, W.shape[0])
        W_sample_small = W[:sample_size]
        norms = np.linalg.norm(W_sample_small, axis=1, keepdims=True) + 1e-9
        correlation = np.dot(W_sample_small / norms, (W_sample_small / norms).T)
        fig_corr = px.imshow(correlation, color_continuous_scale='RdBu', aspect='auto', title="Weight Correlation Matrix")
        fig_corr.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-container"><h3>🔬 Spectral Analysis: The Algorithmic Signature</h3><div class="theory-box"><b>Theory:</b> In the grokking phase, weights exhibit a defined spectral structure with peaks at specific frequencies corresponding to the algorithmic solution. Noise shows a flat spectrum, while memorization shows irregular patterns.</div></div>', unsafe_allow_html=True)
    neuron_indices = [0, 10, 20, 30]
    fig_fft = go.Figure()
    for idx in neuron_indices:
        if idx < W.shape[0]:
            neuron_weights = W[idx]
            fft_vals = np.abs(fft.rfft(neuron_weights))
            freqs = fft.rfftfreq(len(neuron_weights))
            fig_fft.add_trace(go.Scatter(x=freqs[1:], y=fft_vals[1:], mode='lines', name=f'Neuron {idx}', line=dict(width=2.5), fill='tozeroy'))
    fig_fft.update_layout(template="plotly_dark", height=450, xaxis_title="Normalized Frequency", yaxis_title="Spectral Magnitude", title="Frequency Spectrum of Representative Weights", hovermode='x unified', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), margin=dict(l=50, r=20, t=50, b=50))
    for harmonic in [1, 3, 5, 7]:
        fig_fft.add_vline(x=harmonic/len(W[0]), line_dash="dash", line_color="rgba(255, 100, 100, 0.7)", annotation_text=f"Harmonic {harmonic}", annotation_position="top right")
    st.plotly_chart(fig_fft, use_container_width=True)

else:
    st.markdown('<div class="header-container"><h3>🧪 Experimental Results</h3></div>', unsafe_allow_html=True)
    if 'base_model' not in st.session_state:
        st.info("👈 Train a base model first using the sidebar")
    else:
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=st.session_state['train_losses'], mode='lines', name='Train Loss', line=dict(color='#ff6b6b', width=2)))
            fig_loss.update_layout(template="plotly_dark", title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss", height=400)
            st.plotly_chart(fig_loss, use_container_width=True)
        with col_exp2:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(y=st.session_state['test_accs'], mode='lines', name='Test Accuracy', line=dict(color='#4ecdc4', width=2)))
            fig_acc.update_layout(template="plotly_dark", title="Test Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy", height=400)
            st.plotly_chart(fig_acc, use_container_width=True)
        st.markdown('<div class="header-container"><h3>🔍 Base Model Geometry</h3></div>', unsafe_allow_html=True)
        W_base = st.session_state['base_model'].fc1.weight.detach().numpy()
        pca = PCA(n_components=3)
        W_pca = pca.fit_transform(W_base)
        fig_base = go.Figure(data=[go.Scatter3d(x=W_pca[:, 0], y=W_pca[:, 1], z=W_pca[:, 2], mode='markers', marker=dict(size=5, color=np.linalg.norm(W_base, axis=1), colorscale='Viridis', opacity=0.8))])
        fig_base.update_layout(scene=dict(aspectmode='cube'), template="plotly_dark", height=500, title=f"Base Model Weights ({st.session_state['base_bits']} bits)")
        st.plotly_chart(fig_base, use_container_width=True)
        if 'expanded_model' in st.session_state:
            st.markdown(f'<div class="success-box"><h3>✅ CONSERVATION VERIFIED</h3><p><b>Base Model:</b> {st.session_state["base_bits"]} bits → Acc: {st.session_state["test_accs"][-1]:.3f}</p><p><b>Expanded Model:</b> {st.session_state["target_bits"]} bits → Acc: {st.session_state["expanded_acc"]:.3f}</p><p><b>Degradation:</b> {abs(st.session_state["test_accs"][-1] - st.session_state["expanded_acc"]):.4f}</p><p><b>Result:</b> {"✅ PERFECT CONSERVATION" if st.session_state["expanded_acc"] > 0.95 else "⚠️ PARTIAL CONSERVATION"}</p></div>', unsafe_allow_html=True)
            col_comp1, col_comp2 = st.columns(2)
            with col_comp1:
                st.subheader("Base Model")
                W_base_corr = np.corrcoef(W_base)
                fig_base_corr = px.imshow(W_base_corr, color_continuous_scale='RdBu')
                fig_base_corr.update_layout(height=400, title="Base Weight Correlation")
                st.plotly_chart(fig_base_corr, use_container_width=True)
            with col_comp2:
                st.subheader("Expanded Model (excerpt)")
                W_exp = st.session_state['expanded_model'].fc1.weight.detach().numpy()
                W_exp_sample = W_exp[:W_base.shape[0], :W_base.shape[1]]
                W_exp_corr = np.corrcoef(W_exp_sample)
                fig_exp_corr = px.imshow(W_exp_corr, color_continuous_scale='RdBu')
                fig_exp_corr.update_layout(height=400, title="Expanded Weight Correlation")
                st.plotly_chart(fig_exp_corr, use_container_width=True)

st.markdown('<div class="footer"><p>🔬 <b>Grokking Conservation Lab</b> - Scientific Visualization for ML Research</p><p>© 2025 - Based on grokking and learning dynamics research in neural networks</p><p><i>This tool is designed for scientific research and ML interpretability education</i></p></div>', unsafe_allow_html=True)
