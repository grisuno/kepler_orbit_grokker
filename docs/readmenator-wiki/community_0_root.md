# root

*Community 0 | 2 files | cohesion 1.00*

## Definition

This community groups 2 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `GrokkingCaptureWrapper`, `KeplerOrbitPredictor`, `ThermodynamicAnalyzer`, `__init__`, `_capture_phase`, `_create_realtime_chart_with_metrics`, `_detect_phase`, `_initialize_weights`. Core file: `view.py` (14 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:.

## Files

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 11 | yes |
| `view.py` | py | presentation | 14 | yes |

## Key Symbols

- `generate_kepler_orbits` (function, `app.py:22`) `def generate_kepler_orbits(n_samples, noise_level, max_time, seed)` - Generates 2D Keplerian orbit data with enhanced control and quality.
- `KeplerOrbitPredictor` (class, `app.py:67`) `class KeplerOrbitPredictor(Module)` - Optimized MLP for learning physical algorithms with geometric structure
- `__init__` (method, `app.py:70`) `def __init__(self, input_size, hidden_size, output_size)`
- `_initialize_weights` (method, `app.py:82`) `def _initialize_weights(self)` - Weight initialization that favors geometric relationship learning
- `forward` (method, `app.py:89`) `def forward(self, x)`
- `train_until_grok` (method, `app.py:93`) `def train_until_grok(model, X_train, y_train, X_test, y_test, max_epochs, patien` - Adaptive training optimized for physical problems.
- `analyze_geometric_representation` (method, `app.py:198`) `def analyze_geometric_representation(model, X_sample)` - Analyzes whether the model preserves geometric structures
- `expand_model_weights_geometric` (method, `app.py:228`) `def expand_model_weights_geometric(base_model, scale_factor)` - GEOMETRIC EXPANSION FOR PHYSICAL PROBLEMS
- `evaluate_model` (method, `app.py:282`) `def evaluate_model(model, X_test, y_test, model_name, num_examples)` - Evaluates the model and visualizes predictions vs ground truth
- `plot_learning_curves` (method, `app.py:366`) `def plot_learning_curves(history, model_name)` - Visualizes detailed learning curves
- `main` (method, `app.py:391`) `def main()`
- `ThermodynamicAnalyzer` (class, `view.py:40`) `class ThermodynamicAnalyzer` - Analyzes weight space as thermodynamic system
- `compute_metrics` (method, `view.py:44`) `def compute_metrics(weights_data, phase, epoch)` - Calculate complete thermodynamic state
- `visualize_thermal_engine` (method, `view.py:92`) `def visualize_thermal_engine(thermo_history)` - Complete thermal engine visualization
- `GrokkingCaptureWrapper` (class, `view.py:182`) `class GrokkingCaptureWrapper` - Wraps app.py training to capture phase transitions
- `__init__` (method, `view.py:185`) `def __init__(self, model, X_train, y_train, X_test, y_test)`
- `train_with_capture` (method, `view.py:215`) `def train_with_capture(self, max_epochs, snapshot_every)` - Train using EXACT app.py logic with LC and Superposition tracking
- `_detect_phase` (method, `view.py:399`) `def _detect_phase(self, epoch, train_loss, test_loss)` - Detect current training phase
- `_is_loss_dropping_fast` (method, `view.py:411`) `def _is_loss_dropping_fast(self)` - Check if test loss is dropping rapidly
- `_capture_phase` (method, `view.py:419`) `def _capture_phase(self, phase_name, epoch, train_loss, test_loss)` - Capture weight snapshot and thermodynamic state - ALL LAYERS
- `_create_realtime_chart_with_metrics` (method, `view.py:447`) `def _create_realtime_chart_with_metrics(self)` - Create real-time chart with LC and Superposition
- `visualize_3d_weights` (method, `view.py:535`) `def visualize_3d_weights(weights_data, phase_name)` - 3D PCA visualization showing gas/liquid/solid structure
- `visualize_2d_texture` (method, `view.py:638`) `def visualize_2d_texture(weights_data, phase_name)` - 2D weight texture visualization
- `visualize_orbit_predictions` (method, `view.py:691`) `def visualize_orbit_predictions(model, X_test, y_test, num_samples)` - Visualize orbital predictions
- `main` (method, `view.py:754`) `def main()`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 1
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- [INFERRED] shares_context community 0 <-> 1 (strength 0.5): Inferred shared context (layer utility) with no import path between community 0 (root) and community 1 (orphans).

## Risks

- No scoped security, taint, cycle, or layer risks.

## Open Questions

- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `view.py`
