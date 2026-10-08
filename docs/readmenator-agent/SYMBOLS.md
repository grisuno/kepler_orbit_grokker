# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `KeplerOrbitPredictor` | class | `app.py:67` | `class KeplerOrbitPredictor(Module)` |
| `__init__` | method | `app.py:70` | `def __init__(self, input_size, hidden_size, output_size)` |
| `_initialize_weights` | method | `app.py:82` | `def _initialize_weights(self)` |
| `analyze_geometric_representation` | method | `app.py:198` | `def analyze_geometric_representation(model, X_sample)` |
| `evaluate_model` | method | `app.py:282` | `def evaluate_model(model, X_test, y_test, model_name, num_examples)` |
| `expand_model_weights_geometric` | method | `app.py:228` | `def expand_model_weights_geometric(base_model, scale_factor)` |
| `forward` | method | `app.py:89` | `def forward(self, x)` |
| `generate_kepler_orbits` | function | `app.py:22` | `def generate_kepler_orbits(n_samples, noise_level, max_time, seed)` |
| `main` | method | `app.py:391` | `def main()` |
| `plot_learning_curves` | method | `app.py:366` | `def plot_learning_curves(history, model_name)` |
| `train_until_grok` | method | `app.py:93` | `def train_until_grok(model, X_train, y_train, X_test, y_test, max_epochs, patience, initial_lr, min_lr...` |
| `GrokkingCaptureWrapper` | class | `view.py:182` | `class GrokkingCaptureWrapper` |
| `ThermodynamicAnalyzer` | class | `view.py:40` | `class ThermodynamicAnalyzer` |
| `__init__` | method | `view.py:185` | `def __init__(self, model, X_train, y_train, X_test, y_test)` |
| `_capture_phase` | method | `view.py:419` | `def _capture_phase(self, phase_name, epoch, train_loss, test_loss)` |
| `_create_realtime_chart_with_metrics` | method | `view.py:447` | `def _create_realtime_chart_with_metrics(self)` |
| `_detect_phase` | method | `view.py:399` | `def _detect_phase(self, epoch, train_loss, test_loss)` |
| `_is_loss_dropping_fast` | method | `view.py:411` | `def _is_loss_dropping_fast(self)` |
| `compute_metrics` | method | `view.py:44` | `def compute_metrics(weights_data, phase, epoch)` |
| `main` | method | `view.py:754` | `def main()` |
| `train_with_capture` | method | `view.py:215` | `def train_with_capture(self, max_epochs, snapshot_every)` |
| `visualize_2d_texture` | method | `view.py:638` | `def visualize_2d_texture(weights_data, phase_name)` |
| `visualize_3d_weights` | method | `view.py:535` | `def visualize_3d_weights(weights_data, phase_name)` |
| `visualize_orbit_predictions` | method | `view.py:691` | `def visualize_orbit_predictions(model, X_test, y_test, num_samples)` |
| `visualize_thermal_engine` | method | `view.py:92` | `def visualize_thermal_engine(thermo_history)` |
