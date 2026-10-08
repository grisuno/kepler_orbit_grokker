# API

## app.py

### generate_kepler_orbits (function) `def generate_kepler_orbits(n_samples, noise_level, max_time, seed)`
- Defined: `app.py:22`
- Doc: Generates 2D Keplerian orbit data with enhanced control and quality.
- Imported by: `view.py`

### train_until_grok (method) `def train_until_grok(model, X_train, y_train, X_test, y_test, max_epochs, patience, initial_lr, min_lr, weight_decay, grok_threshold)`
- Defined: `app.py:93`
- Doc: Adaptive training optimized for physical problems.
- Imported by: `view.py`

### analyze_geometric_representation (method) `def analyze_geometric_representation(model, X_sample)`
- Defined: `app.py:198`
- Doc: Analyzes whether the model preserves geometric structures
- Imported by: `view.py`

### expand_model_weights_geometric (method) `def expand_model_weights_geometric(base_model, scale_factor)`
- Defined: `app.py:228`
- Doc: GEOMETRIC EXPANSION FOR PHYSICAL PROBLEMS 
- Imported by: `view.py`

### evaluate_model (method) `def evaluate_model(model, X_test, y_test, model_name, num_examples)`
- Defined: `app.py:282`
- Doc: Evaluates the model and visualizes predictions vs ground truth
- Imported by: `view.py`

### plot_learning_curves (method) `def plot_learning_curves(history, model_name)`
- Defined: `app.py:366`
- Doc: Visualizes detailed learning curves
- Imported by: `view.py`

### main (method) `def main()`
- Defined: `app.py:391`
- Imported by: `view.py`

### __init__ (method) `def __init__(self, input_size, hidden_size, output_size)`
- Defined: `app.py:70`
- Imported by: `view.py`

### _initialize_weights (method) `def _initialize_weights(self)`
- Defined: `app.py:82`
- Doc: Weight initialization that favors geometric relationship learning
- Imported by: `view.py`

### forward (method) `def forward(self, x)`
- Defined: `app.py:89`
- Imported by: `view.py`

## view.py

### visualize_3d_weights (method) `def visualize_3d_weights(weights_data, phase_name)`
- Defined: `view.py:535`
- Doc: 3D PCA visualization showing gas/liquid/solid structure
- Depends on: `app.py`

### visualize_2d_texture (method) `def visualize_2d_texture(weights_data, phase_name)`
- Defined: `view.py:638`
- Doc: 2D weight texture visualization
- Depends on: `app.py`

### visualize_orbit_predictions (method) `def visualize_orbit_predictions(model, X_test, y_test, num_samples)`
- Defined: `view.py:691`
- Doc: Visualize orbital predictions
- Depends on: `app.py`

### main (method) `def main()`
- Defined: `view.py:754`
- Depends on: `app.py`

### compute_metrics (method) `def compute_metrics(weights_data, phase, epoch)`
- Defined: `view.py:44`
- Doc: Calculate complete thermodynamic state
- Depends on: `app.py`

### visualize_thermal_engine (method) `def visualize_thermal_engine(thermo_history)`
- Defined: `view.py:92`
- Doc: Complete thermal engine visualization
- Depends on: `app.py`

### __init__ (method) `def __init__(self, model, X_train, y_train, X_test, y_test)`
- Defined: `view.py:185`
- Depends on: `app.py`

### train_with_capture (method) `def train_with_capture(self, max_epochs, snapshot_every)`
- Defined: `view.py:215`
- Doc: Train using EXACT app.py logic with LC and Superposition tracking
- Depends on: `app.py`

### _detect_phase (method) `def _detect_phase(self, epoch, train_loss, test_loss)`
- Defined: `view.py:399`
- Doc: Detect current training phase
- Depends on: `app.py`

### _is_loss_dropping_fast (method) `def _is_loss_dropping_fast(self)`
- Defined: `view.py:411`
- Doc: Check if test loss is dropping rapidly
- Depends on: `app.py`

### _capture_phase (method) `def _capture_phase(self, phase_name, epoch, train_loss, test_loss)`
- Defined: `view.py:419`
- Doc: Capture weight snapshot and thermodynamic state - ALL LAYERS
- Depends on: `app.py`

### _create_realtime_chart_with_metrics (method) `def _create_realtime_chart_with_metrics(self)`
- Defined: `view.py:447`
- Doc: Create real-time chart with LC and Superposition
- Depends on: `app.py`
