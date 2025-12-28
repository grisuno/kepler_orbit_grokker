```text
INITIATING GEOMETRIC GROKING POC FOR KEPLERIAN ORBITS
================================================================================

GENERATING KEPLERIAN ORBIT DATA...
Generated data: 2000 total samples
Train: 1600, Test: 400

TRAINING BASE MODEL (128 hidden units)...
INITIATING ADAPTIVE TRAINING FOR PHYSICAL GROKKING...
Parameters: max_epochs=25000, initial_lr=0.003, weight_decay=5e-06
TRAINING FOR GROKING:  46%|███████▉         | 11588/25000 [00:46<01:16, 175.20it/s, train_loss=0.000005, test_loss=0.000050, best_test=0.000050, lr=7.50e-04]
PHYSICAL GROKING ACHIEVED at epoch 11595!
Test loss: 0.00004997 < threshold 5.00e-05
TRAINING FOR GROKING:  46%|███████▉         | 11595/25000 [00:46<00:53, 249.25it/s, train_loss=0.000005, test_loss=0.000050, best_test=0.000050, lr=7.50e-04]

ANALYZING BASE MODEL GEOMETRIC REPRESENTATION...
Angles H1: 0.5937
Angles H2: 0.5672
Distance preservation: 0.9786

GENERATING VISUALIZATIONS...

EVALUATING BASE MODEL...
Base_Model - EVALUATION:
Total MSE: 0.00004997
MSE X: 0.00005004, MSE Y: 0.00004990
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Plot saved: orbit_predictions_base_model.png

EXPANDING MODEL WITH GEOMETRIC TRANSFER...
GEOMETRIC EXPANSION CORRECTED: 128 → 256

ANALYZING EXPANDED GEOMETRIC REPRESENTATION...
Angles H1: 0.5933
Angles H2: 0.5712
Distance preservation: 0.9789

EVALUATING EXPANDED MODEL (ZERO-SHOT TRANSFER)...
Expanded_Model - EVALUATION:
Total MSE: 0.09889212
MSE X: 0.11047431, MSE Y: 0.08730994
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Plot saved: orbit_predictions_expanded_model.png

GENERATING SCALED DATA...

EVALUATING ON SCALED PROBLEM...
Scaled_Model - EVALUATION:
Total MSE: 0.07849496
MSE X: 0.08474111, MSE Y: 0.07224881
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Ignoring fixed y limits to fulfill fixed data aspect with adjustable data limits.
Plot saved: orbit_predictions_scaled_model.png

================================================================================
FINAL RESULTS
================================================================================
Base Model (128 units)              | MSE: 0.00004997 | Grokking: YES
Expanded Model (256 units)          | MSE: 0.09889212 | Zero-shot: WARNING
Scaled Model (complex data)         | MSE: 0.07849496 | Transfer: WARNING
================================================================================

SUCCESS ANALYSIS:
Physical algorithm grokking: ACHIEVED
Zero-shot transfer: MODERATE
Geometric preservation: MAINTAINED
Scaling to complex problems: ACCEPTABLE

OVERALL RESULT: SIGNIFICANT PROGRESS!

Saved plots:
   - orbit_predictions_base_model.png
   - orbit_predictions_expanded_model.png
   - orbit_predictions_scaled_model.png
   - learning_curves_base_model.png

```
