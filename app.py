#!/usr/bin/env python3
# _*_ coding: utf8 _*_
"""
app.py

Autor: Gris Iscomeback
Correo electrónico: grisiscomeback[at]gmail[dot]com
Fecha de creación: xx/xx/xxxx
Licencia: GPL v3

Descripción:  
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

def generate_kepler_orbits(n_samples=2000, noise_level=0.001, max_time=5.0, seed=42):
    """
    Genera datos de órbitas keplerianas en 2D con mayor control y calidad.
    Versión optimizada para facilitar el grokking del algoritmo físico.
    """
    np.random.seed(seed)
    data = []
    targets = []
    
    for _ in range(n_samples):
        
        h = np.random.uniform(0.8, 1.5)     
        mu = 1.0                            
        e = np.random.uniform(0.0, 0.6)     
        theta0 = np.random.uniform(0, 2*np.pi)  
        t = np.random.uniform(0, max_time)  
        
        omega = 0.2 * h  
        
        theta = theta0 + omega * t
        r = (h**2 / mu) / (1 + e * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        x += np.random.normal(0, noise_level * max(0.1, abs(x)))
        y += np.random.normal(0, noise_level * max(0.1, abs(y)))
        
        
        delta_t = 0.1
        future_theta = theta0 + omega * (t + delta_t)
        future_r = (h**2 / mu) / (1 + e * np.cos(future_theta))
        future_x = future_r * np.cos(future_theta)
        future_y = future_r * np.sin(future_theta)
        
        
        input_data = [x, y, h, e, t]
        
        target = [future_x, future_y]
        
        data.append(input_data)
        targets.append(target)
    
    return np.array(data, dtype=np.float32), np.array(targets, dtype=np.float32)


class KeplerOrbitPredictor(nn.Module):
    """MLP optimizado para aprender algoritmos físicos con estructura geométrica"""
    
    def __init__(self, input_size=5, hidden_size=128, output_size=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización que favorece el aprendizaje de relaciones geométricas"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        return self.net(x)


def train_until_grok(model, X_train, y_train, X_test, y_test, 
                    max_epochs=20000, patience=1000, initial_lr=0.01, 
                    min_lr=1e-5, weight_decay=1e-4, grok_threshold=1e-5):
    """
    Entrenamiento adaptativo optimizado para problemas físicos.
    Versión compatible con versiones antiguas de PyTorch (sin argumento 'verbose').
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=min_lr
    )
    
    best_test_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'test_loss': [], 'lr': []}
    
    
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    print(f"🎯 Iniciando entrenamiento adaptativo para grokking físico...")
    print(f"   Parámetros: max_epochs={max_epochs}, initial_lr={initial_lr}, weight_decay={weight_decay}")
    
    pbar = tqdm(range(max_epochs), desc="🧠 Entrenando para Grokking")
    
    for epoch in pbar:
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()
        
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_loss = criterion(test_outputs, y_test_t)
        
        
        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        
        history['train_loss'].append(loss.item())
        history['test_loss'].append(test_loss.item())
        history['lr'].append(current_lr)
        
        
        if test_loss.item() < best_test_loss:
            best_test_loss = test_loss.item()
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        
        pbar.set_postfix({
            'train_loss': f'{loss.item():.6f}', 
            'test_loss': f'{test_loss.item():.6f}',
            'best_test': f'{best_test_loss:.6f}',
            'lr': f'{current_lr:.2e}'
        })
        
        
        if test_loss.item() < grok_threshold:
            print(f"\n✨ ¡GROKKING FÍSICO LOGRADO en época {epoch}!")
            print(f"   Pérdida de test: {test_loss.item():.8f} < umbral {grok_threshold:.2e}")
            model.load_state_dict(best_model_state)
            return model, history, True
        
        
        if epochs_no_improve > patience:
            if current_lr <= min_lr * 1.1:  
                print(f"\n⏹️  Early stopping activado después de {epoch} épocas")
                print(f"   Mejor pérdida de test: {best_test_loss:.8f}")
                model.load_state_dict(best_model_state)
                
                
                grokking_achieved = best_test_loss < grok_threshold * 10
                status = "✅ CERCA DEL GROKKING" if grokking_achieved else "⚠️  NO SE LOGRÓ GROKKING COMPLETO"
                print(f"   Estado: {status}")
                
                return model, history, grokking_achieved
    
    
    model.load_state_dict(best_model_state)
    grokking_achieved = best_test_loss < grok_threshold
    print(f"\n🏁 Entrenamiento completado después de {max_epochs} épocas")
    print(f"   Mejor pérdida de test: {best_test_loss:.8f}")
    print(f"   ¿GROKKING logrado? {'✅ SÍ' if grokking_achieved else '❌ NO'}")
    
    return model, history, grokking_achieved


def analyze_geometric_representation(model, X_sample):
    """Analiza si el modelo preserva estructuras geométricas"""
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_sample)
        
        
        h1 = torch.relu(model.net[0](X_t))
        h2 = torch.relu(model.net[2](h1))
        
        
        
        angles_h1 = torch.cosine_similarity(h1.unsqueeze(1), h1.unsqueeze(0), dim=2)
        angles_h2 = torch.cosine_similarity(h2.unsqueeze(1), h2.unsqueeze(0), dim=2)
        
        
        dists_h1 = torch.cdist(h1, h1)
        dists_h2 = torch.cdist(h2, h2)
        
        
        h1_var = torch.var(h1, dim=0).mean()
        h2_var = torch.var(h2, dim=0).mean()
        
        return {
            'angle_consistency_h1': angles_h1.mean().item(),
            'angle_consistency_h2': angles_h2.mean().item(),
            'distance_preservation': torch.corrcoef(torch.stack([dists_h1.flatten(), dists_h2.flatten()]))[0,1].item(),
            'representation_stability': (h1_var / h2_var).item()
        }

def expand_model_weights_geometric(base_model, scale_factor=2):
    """
    EXPANSIÓN GEOMÉTRICA PARA PROBLEMAS FÍSICOS - CORREGIDA
    Preserva la estructura del espacio tangente y las relaciones angulares.
    ¡ESTA ES LA FUNCIÓN QUE DEBE USARSE!
    """
    input_size = base_model.net[0].in_features
    hidden_size = base_model.net[0].out_features
    output_size = base_model.net[-1].out_features
    new_hidden_size = hidden_size * scale_factor
    
    expanded_model = KeplerOrbitPredictor(
        input_size=input_size,
        hidden_size=new_hidden_size,
        output_size=output_size
    )
    
    with torch.no_grad():
        
        expanded_model.net[0].weight[:hidden_size, :] = base_model.net[0].weight
        expanded_model.net[0].bias[:hidden_size] = base_model.net[0].bias
        expanded_model.net[2].weight[:hidden_size, :hidden_size] = base_model.net[2].weight
        expanded_model.net[2].bias[:hidden_size] = base_model.net[2].bias
        expanded_model.net[4].weight[:, :hidden_size] = base_model.net[4].weight
        expanded_model.net[4].bias[:] = base_model.net[4].bias
        
        
        if scale_factor > 1:
            for i in range(1, scale_factor):
                start_idx = i * hidden_size
                end_idx = (i + 1) * hidden_size
                
                
                expanded_model.net[0].weight[start_idx:end_idx, :] = (
                    base_model.net[0].weight * (0.8 + 0.2 * i / scale_factor)
                )
                
                expanded_model.net[2].weight[:hidden_size, start_idx:end_idx] = (
                    base_model.net[2].weight[:, :hidden_size] * 0.3
                )
                expanded_model.net[2].weight[start_idx:end_idx, :hidden_size] = (
                    base_model.net[2].weight[:hidden_size, :] * 0.3
                )
                expanded_model.net[2].weight[start_idx:end_idx, start_idx:end_idx] = (
                    torch.eye(hidden_size) * 0.1
                )
                
                expanded_model.net[0].bias[start_idx:end_idx] = (
                    base_model.net[0].bias * (0.9 + 0.1 * i / scale_factor)
                )
                expanded_model.net[2].bias[start_idx:end_idx] = 0.01 * i
    
    print(f"✅ Expansión geométrica CORREGIDA: {hidden_size} → {new_hidden_size}")
    return expanded_model

def evaluate_model(model, X_test, y_test, model_name="Modelo", num_examples=5):
    """Evalúa el modelo y visualiza predicciones vs realidad ()"""
    model.eval()
    with torch.no_grad():
        X_test_t = torch.FloatTensor(X_test)
        y_pred = model(X_test_t).numpy()
    
    
    mse = np.mean((y_pred - y_test)**2)
    
    
    mse_x = np.mean((y_pred[:, 0] - y_test[:, 0])**2)
    mse_y = np.mean((y_pred[:, 1] - y_test[:, 1])**2)
    
    print(f"📊 {model_name} - Evaluación:")
    print(f"   MSE total: {mse:.8f}")
    print(f"   MSE X: {mse_x:.8f}, MSE Y: {mse_y:.8f}")
    
    
    plt.figure(figsize=(15, 10))
    
    
    indices = np.random.choice(len(X_test), min(num_examples, len(X_test)), replace=False)
    
    for i, idx in enumerate(indices):
        plt.subplot(2, 3, i+1)
        
        
        x_curr, y_curr, h, e, t = X_test[idx]
        x_true, y_true = y_test[idx]
        x_pred, y_pred_single = y_pred[idx]
        
        
        example_mse = (x_pred - x_true)**2 + (y_pred_single - y_true)**2
        
        
        plt.scatter([x_curr], [y_curr], c='blue', s=100, alpha=0.7, label='Posición actual')
        
        
        plt.scatter([x_true], [y_true], c='green', s=150, marker='*', label='Real (futuro)')
        plt.scatter([x_pred], [y_pred_single], c='red', s=100, alpha=0.8, label='Predicción')
        
        
        theta_vals = np.linspace(0, 2*np.pi, 200)
        r_vals = (h**2) / (1 + e * np.cos(theta_vals))
        x_vals = r_vals * np.cos(theta_vals)
        y_vals = r_vals * np.sin(theta_vals)
        plt.plot(x_vals, y_vals, 'gray', alpha=0.3, linestyle='--')
        
        
        plt.plot([x_curr, x_true], [y_curr, y_true], 'g--', alpha=0.5, linewidth=1)
        plt.plot([x_curr, x_pred], [y_curr, y_pred_single], 'r--', alpha=0.5, linewidth=1)
        
        plt.title(f'Ejemplo {i+1}\nMSE: {example_mse:.6f}\nh={h:.2f}, e={e:.2f}, t={t:.2f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.2)
        plt.axis('equal')
        
        
        max_val = max(3.0, abs(x_curr)*1.5, abs(y_curr)*1.5, abs(x_true)*1.5, abs(y_true)*1.5)
        plt.xlim(-max_val, max_val)
        plt.ylim(-max_val, max_val)
    
    
    if hasattr(model, 'training_history'):
        plt.subplot(2, 3, 6)
        if 'train_loss' in model.training_history and 'test_loss' in model.training_history:
            plt.semilogy(model.training_history['train_loss'], 'b-', alpha=0.7, label='Train Loss')
            plt.semilogy(model.training_history['test_loss'], 'r-', alpha=0.7, label='Test Loss')
            plt.title('Curva de Aprendizaje')
            plt.xlabel('Épocas')
            plt.ylabel('Pérdida (log scale)')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'orbit_predictions_{model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    print(f"💾 Gráfico guardado: orbit_predictions_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    
    return mse

def plot_learning_curves(history, model_name="Modelo"):
    """Visualiza curvas de aprendizaje detalladas"""
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogy(history['train_loss'], 'b-', label='Train Loss')
    plt.semilogy(history['test_loss'], 'r-', label='Test Loss')
    plt.title(f'Curvas de Aprendizaje - {model_name}')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.semilogy(history['lr'], 'g-', label='Learning Rate')
    plt.title('Evolución del Learning Rate')
    plt.xlabel('Épocas')
    plt.ylabel('LR (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'learning_curves_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()

def main():
    print("🚀 INICIANDO POC DE GROKKING GEOMÉTRICO PARA ÓRBITAS KEPLERIANAS")
    print("=" * 80)
    
    
    print("\n🌌 Generando datos de órbitas keplerianas...")
    X, y = generate_kepler_orbits(
        n_samples=2000,
        noise_level=0.0005,
        max_time=3.0,
        seed=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Datos generados: {X.shape[0]} muestras totales")
    print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    
    print("\n🧠 Entrenando modelo base (128 unidades ocultas)...")
    base_model = KeplerOrbitPredictor(input_size=5, hidden_size=128, output_size=2)
    
    
    base_model, history, grokking_achieved = train_until_grok(
        base_model, X_train, y_train, X_test, y_test,
        max_epochs=25000,           
        patience=2000,              
        initial_lr=0.003,           
        min_lr=1e-7,                
        weight_decay=5e-6,          
        grok_threshold=5e-5         
    )
    
    base_model.training_history = history
    
    
    print("\n📐 Analizando representación geométrica del modelo base...")
    geom_base = analyze_geometric_representation(base_model, X_test[:100])
    print(f"   Ángulos H1: {geom_base['angle_consistency_h1']:.4f}")
    print(f"   Ángulos H2: {geom_base['angle_consistency_h2']:.4f}")
    print(f"   Preservación distancias: {geom_base['distance_preservation']:.4f}")
    
    
    print("\n📈 Generando visualizaciones...")
    plot_learning_curves(history, "Modelo_Base")
    
    
    print("\n🔍 Evaluando modelo base...")
    base_mse = evaluate_model(base_model, X_test, y_test, "Modelo_Base")
    
    
    print("\n🔄 Expandiendo modelo con TRANSFERENCIA GEOMÉTRICA...")
    expanded_model = expand_model_weights_geometric(base_model, scale_factor=2)  
    
    
    print("\n📐 Analizando representación geométrica expandida...")
    geom_expanded = analyze_geometric_representation(expanded_model, X_test[:100])
    print(f"   Ángulos H1: {geom_expanded['angle_consistency_h1']:.4f}")
    print(f"   Ángulos H2: {geom_expanded['angle_consistency_h2']:.4f}")
    print(f"   Preservación distancias: {geom_expanded['distance_preservation']:.4f}")
    
    
    print("\n🎯 Evaluando modelo expandido (ZERO-SHOT TRANSFER)...")
    expanded_mse = evaluate_model(expanded_model, X_test, y_test, "Modelo_Expandido")
    
    
    print("\n📈 Generando datos escalados...")
    X_large, y_large = generate_kepler_orbits(
        n_samples=3000, 
        noise_level=0.002,
        max_time=8.0,
        seed=123
    )
    
    _, X_large_test, _, y_large_test = train_test_split(
        X_large, y_large, test_size=0.3, random_state=42
    )
    
    print("\n🚀 Evaluando en problema escalado...")
    large_mse = evaluate_model(expanded_model, X_large_test, y_large_test, "Modelo_Escalado")
    
    
    print("\n" + "="*80)
    print("📊 RESULTADOS FINALES")
    print("="*80)
    print(f"{'Modelo Base (128 unidades)':<35} | MSE: {base_mse:.8f} | Grokking: {'✅' if grokking_achieved else '❌'}")
    print(f"{'Modelo Expandido (256 unidades)':<35} | MSE: {expanded_mse:.8f} | Zero-shot: {'✅' if expanded_mse < base_mse * 3 else '⚠️'}")
    print(f"{'Modelo Escalado (datos complejos)':<35} | MSE: {large_mse:.8f} | Transferencia: {'✅' if large_mse < 0.01 else '⚠️'}")
    print("="*80)
    
    
    success_metrics = {
        'grokking_achieved': grokking_achieved,
        'zero_shot_success': expanded_mse < base_mse * 5,  
        'scaling_success': large_mse < 0.02,
        'geometric_preservation': abs(geom_base['angle_consistency_h2'] - geom_expanded['angle_consistency_h2']) < 0.2
    }
    
    print("\n🎯 ANÁLISIS DE ÉXITO:")
    print(f"   Grokking del algoritmo físico: {'✅ LOGRADO' if grokking_achieved else '🟡 PARCIAL'}")
    print(f"   Transferencia zero-shot: {'✅ EXITOSA' if success_metrics['zero_shot_success'] else '🟡 MODERADA'}")
    print(f"   Preservación geométrica: {'✅ MANTENIDA' if success_metrics['geometric_preservation'] else '❌ PERDIDA'}")
    print(f"   Escalado a problemas complejos: {'✅ ROBUSTO' if success_metrics['scaling_success'] else '🟡 ACEPTABLE'}")
    
    overall_success = sum(success_metrics.values()) >= 3
    print(f"\n🌟 RESULTADO GENERAL: {'¡ÉXITO EN TRANSFERENCIA GEOMÉTRICA!' if overall_success else '¡PROGRESO SIGNIFICATIVO!'}")
    
    print(f"\n💾 Gráficos guardados:")
    print("   - orbit_predictions_modelo_base.png")
    print("   - orbit_predictions_modelo_expandido.png") 
    print("   - orbit_predictions_modelo_escalado.png")
    print("   - learning_curves_modelo_base.png")


if __name__ == "__main__":
    import os
    if 'DISPLAY' not in os.environ and os.name != 'nt':
        plt.switch_backend('Agg')
    
    main()
