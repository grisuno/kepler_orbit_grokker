# API

## app.py
Imported by: `view.py`
- `generate_kepler_orbits` (function) `app.py:22` `def generate_kepler_orbits(n_samples, noise_level, max_time, seed)` -- Generates 2D Keplerian orbit data with enhanced control and quality.
- `KeplerOrbitPredictor.__init__` (method) `app.py:70` `def __init__(self, input_size, hidden_size, output_size)`
- `KeplerOrbitPredictor.forward` (method) `app.py:89` `def forward(self, x)`
- `KeplerOrbitPredictor.train_until_grok` (method) `app.py:93` `def train_until_grok(model, X_train, y_train, X_test, y_test, max_epochs, patience, initial_lr, min_lr...` -- Adaptive training optimized for physical problems.
- `KeplerOrbitPredictor.analyze_geometric_representation` (method) `app.py:198` `def analyze_geometric_representation(model, X_sample)` -- Analyzes whether the model preserves geometric structures
- `KeplerOrbitPredictor.expand_model_weights_geometric` (method) `app.py:228` `def expand_model_weights_geometric(base_model, scale_factor)` -- GEOMETRIC EXPANSION FOR PHYSICAL PROBLEMS Preserves tangent space structure and angular relationships.
- `KeplerOrbitPredictor.evaluate_model` (method) `app.py:282` `def evaluate_model(model, X_test, y_test, model_name, num_examples)` -- Evaluates the model and visualizes predictions vs ground truth
- `KeplerOrbitPredictor.plot_learning_curves` (method) `app.py:366` `def plot_learning_curves(history, model_name)` -- Visualizes detailed learning curves
- `KeplerOrbitPredictor.main` (method) `app.py:391` `def main()`

## view.py
Depends on: `app.py`
- `ThermodynamicAnalyzer.compute_metrics` (method) `view.py:44` `def compute_metrics(weights_data, phase, epoch)` -- Calculate complete thermodynamic state
- `ThermodynamicAnalyzer.visualize_thermal_engine` (method) `view.py:92` `def visualize_thermal_engine(thermo_history)` -- Complete thermal engine visualization
- `GrokkingCaptureWrapper.__init__` (method) `view.py:185` `def __init__(self, model, X_train, y_train, X_test, y_test)`
- `GrokkingCaptureWrapper.train_with_capture` (method) `view.py:215` `def train_with_capture(self, max_epochs, snapshot_every)` -- Train using EXACT app.py logic with LC and Superposition tracking
- `GrokkingCaptureWrapper.visualize_3d_weights` (method) `view.py:535` `def visualize_3d_weights(weights_data, phase_name)` -- 3D PCA visualization showing gas/liquid/solid structure
- `GrokkingCaptureWrapper.visualize_2d_texture` (method) `view.py:638` `def visualize_2d_texture(weights_data, phase_name)` -- 2D weight texture visualization
- `GrokkingCaptureWrapper.visualize_orbit_predictions` (method) `view.py:691` `def visualize_orbit_predictions(model, X_test, y_test, num_samples)` -- Visualize orbital predictions
- `GrokkingCaptureWrapper.main` (method) `view.py:754` `def main()`
