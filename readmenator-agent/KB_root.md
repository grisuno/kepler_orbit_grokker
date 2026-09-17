# Subsystem: root

## app.py
- Layer: utility
- Doc: app.py  Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licenci
- Language: py
- Symbols:
  - `generate_kepler_orbits` (function, line 22) `def generate_kepler_orbits(n_samples, noise_level, max_time, seed)`
  - `KeplerOrbitPredictor` (class, line 67) `class KeplerOrbitPredictor(Module)`
  - `train_until_grok` (method, line 93) `def train_until_grok(model, X_train, y_train, X_test, y_test, max_epochs, patience, initial_lr, min_lr, weight_decay, grok_threshold)`
  - `analyze_geometric_representation` (method, line 198) `def analyze_geometric_representation(model, X_sample)`
  - `expand_model_weights_geometric` (method, line 228) `def expand_model_weights_geometric(base_model, scale_factor)`
  - `evaluate_model` (method, line 282) `def evaluate_model(model, X_test, y_test, model_name, num_examples)`
  - `plot_learning_curves` (method, line 366) `def plot_learning_curves(history, model_name)`
  - `main` (method, line 391) `def main()`
  - `__init__` (method, line 70) `def __init__(self, input_size, hidden_size, output_size)`
  - `_initialize_weights` (method, line 82) `def _initialize_weights(self)`
  - `forward` (method, line 89) `def forward(self, x)`
- Imported by: `view.py`

## install.sh
- Layer: utility
- Language: sh

## view.py
- Layer: presentation
- Doc: view.py - COMPLETE GROKKING PHASE TRANSITION VISUALIZER
- Language: py
- Symbols:
  - `ThermodynamicAnalyzer` (class, line 40) `class ThermodynamicAnalyzer`
  - `GrokkingCaptureWrapper` (class, line 182) `class GrokkingCaptureWrapper`
  - `visualize_3d_weights` (method, line 535) `def visualize_3d_weights(weights_data, phase_name)`
  - `visualize_2d_texture` (method, line 638) `def visualize_2d_texture(weights_data, phase_name)`
  - `visualize_orbit_predictions` (method, line 691) `def visualize_orbit_predictions(model, X_test, y_test, num_samples)`
  - `main` (method, line 754) `def main()`
  - `compute_metrics` (method, line 44) `def compute_metrics(weights_data, phase, epoch)`
  - `visualize_thermal_engine` (method, line 92) `def visualize_thermal_engine(thermo_history)`
  - `__init__` (method, line 185) `def __init__(self, model, X_train, y_train, X_test, y_test)`
  - `train_with_capture` (method, line 215) `def train_with_capture(self, max_epochs, snapshot_every)`
  - `_detect_phase` (method, line 399) `def _detect_phase(self, epoch, train_loss, test_loss)`
  - `_is_loss_dropping_fast` (method, line 411) `def _is_loss_dropping_fast(self)`
  - `_capture_phase` (method, line 419) `def _capture_phase(self, phase_name, epoch, train_loss, test_loss)`
  - `_create_realtime_chart_with_metrics` (method, line 447) `def _create_realtime_chart_with_metrics(self)`
- Depends on: `app.py`
