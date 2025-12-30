# Kepler Orbit Grokker
Zero-shot transfer of learned physical laws via geometric weight expansion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18072859.svg)](https://doi.org/10.5281/zenodo.18072859)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)


## What This Does

Train a small neural network on Keplerian orbital dynamics until it "groks" the underlying physical law. Then expand the model to a larger architecture by copying weights into a larger matrix using geometric-preserving initialization. The expanded model demonstrates zero-shot transfer of the learned orbital algorithm—with no additional training.

Key result: The model groks the exact analytical solution of the two-body problem and preserves its internal geometric representation after expansion, even though absolute prediction error increases due to interference from unused dimensions.

## Results

**Base Model (128 units)**
- MSE: 4.999×10⁻⁵
- Status: ✅ Grokking achieved

**Expanded Model (256 units)**
- MSE: 0.240
- Status: ⚠️ Zero-shot transfer (geometry preserved)

**Scaled Problem**
- MSE: 0.762
- Status: ⚠️ Generalization to complex orbits

**Geometric Consistency (before vs. after expansion):**
- Angle consistency (H1): 0.6492 → 0.6489
- Angle consistency (H2): 0.6215 → 0.6241
- Distance preservation: 0.9828 → 0.9835

*Interpretation: The internal geometric structure of the learned algorithm is preserved under expansion—even when absolute output error rises. This confirms that grokking encodes physics as invariant geometric primitives, not just functional mappings.*

## Method

1. **Train base model**: MLP (2 hidden layers, ReLU) on synthetic Keplerian orbit data until MSE < 5×10⁻⁵ (grokking threshold).

2. **Geometric weight expansion**: 
   - For a weight matrix W ∈ ℝ^(d×n), construct W' ∈ ℝ^(2d×2n):
   - Copy W into the upper-left block
   - Initialize new blocks using correlated replicas scaled by physical priors
   - Preserve angular relationships and manifold structure of the latent space

3. **Evaluate**: Test expanded model immediately, with zero gradient updates.

## Task Definition

We solve the Keplerian orbit prediction problem:

Given current position (x, y), angular momentum h, eccentricity e, and time t, predict future position (x', y') under the law:

r(θ) = h²/μ / (1 + e·cosθ), θ(t) = θ₀ + ωt

This is a continuous, differentiable, algorithmically compact physical law—ideal for grokking.

## Why It Works (in Theory)

When a network groks a physical law with analytical structure, it crystallizes a geometric representation of that law in its weight space. This representation:
- Lives on a low-dimensional manifold embedded in the hidden layers
- Is invariant under certain linear transformations (rotations, scalings)
- Can be expanded by "splitting" neurons while preserving angular relationships

The geometric initialization ensures that new neurons inherit the coordinate system of the original algorithmic primitive.

## Limitations

- Requires initial grokking (can take thousands of epochs)
- Works for tasks with compact algorithmic structure
- Preserves fixed subcircuits, not arbitrary scaling
- Error increases if new dimensions are not masked during evaluation
- Numerical precision limits extreme scales

## Related Work

- **algebra-de-grok**: Zero-shot transfer of subset parity tasks via weight expansion
- **SWAN-Phoenix-Rising**: Applied same method to medical classification (AUROC > 0.99)

## Technical Implementation

# Base training
model = KeplerOrbitGrokker(hidden_dim=128)
train_until_grokking(model, lr=0.003, weight_decay=5e-6)

# Geometric expansion
expanded_model = expand_model_weights_geometric(model, factor=2)

# Zero-shot evaluation
predictions = expanded_model(test_data)  # No fine-tuning

## Citation

@software{kepler_grokker_2025,
  author = {grisun0},
  title = {Kepler Orbit Grokker: Geometric Transfer of Physical Laws},
  year = {2025},
  url = {https://github.com/grisuno/kepler_orbit_grokker}
}

## License

AGPL v3

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
