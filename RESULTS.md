```text
🚀 INICIANDO POC DE GROKKING GEOMÉTRICO PARA ÓRBITAS KEPLERIANAS```🚀 INICIANDO POC DE GROKKING GEOMÉTRICO PARA ÓRBITAS KEPLERIANAS==============

🌌 Generando datos de órbitas keplerianas...
📊 Datos generados: 2000 muestras totales
   Train: 1600, Test: 400

🧠 Entrenando modelo base (128 unidades ocultas)...
🎯 Iniciando entrenamiento adaptativo para grokking físico...
   Parámetros: max_epochs=25000, initial_lr=0.003, weight_decay=5e-06
🧠 Entrenando para Grokking:  42%|████▎     | 10625/25000 [00:49<01:18, 182.56it/s, train_loss=0.000005, test_loss=0.000050, best_test=0.000050, lr=1.50e-03]
✨ ¡GROKKING FÍSICO LOGRADO en época 10640!
   Pérdida de test: 0.00004999 < umbral 5.00e-05
🧠 Entrenando para Grokking:  43%|████▎     | 10640/25000 [00:49<01:06, 214.74it/s, train_loss=0.000005, test_loss=0.000050, best_test=0.000050, lr=1.50e-03]

📐 Analizando representación geométrica del modelo base...
   Ángulos H1: 0.6492
   Ángulos H2: 0.6215
   Preservación distancias: 0.9828

📈 Generando visualizaciones...

🔍 Evaluando modelo base...
📊 Modelo_Base - Evaluación:
   MSE total: 0.00004999
   MSE X: 0.00005368, MSE Y: 0.00004630
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
💾 Gráfico guardado: orbit_predictions_modelo_base.png

🔄 Expandiendo modelo con TRANSFERENCIA GEOMÉTRICA...
✅ Expansión geométrica CORREGIDA: 128 → 256

📐 Analizando representación geométrica expandida...
   Ángulos H1: 0.6489
   Ángulos H2: 0.6241
   Preservación distancias: 0.9835

🎯 Evaluando modelo expandido (ZERO-SHOT TRANSFER)...
📊 Modelo_Expandido - Evaluación:
   MSE total: 0.24007955
   MSE X: 0.07884274, MSE Y: 0.40131637
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
💾 Gráfico guardado: orbit_predictions_modelo_expandido.png

📈 Generando datos escalados...

🚀 Evaluando en problema escalado...
📊 Modelo_Escalado - Evaluación:
   MSE total: 0.76210976
   MSE X: 0.17829686, MSE Y: 1.34592259
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
💾 Gráfico guardado: orbit_predictions_modelo_escalado.png

================================================================================
📊 RESULTADOS FINALES
================================================================================
Modelo Base (128 unidades)          | MSE: 0.00004999 | Grokking: ✅
Modelo Expandido (256 unidades)     | MSE: 0.24007955 | Zero-shot: ⚠
Modelo Escalado (datos complejos)   | MSE: 0.76210976 | Transferencia: ⚠
================================================================================

🎯 ANÁLISIS DE ÉXITO:
   Grokking del algoritmo físico: ✅ LOGRADO
   Transferencia zero-shot: 🟡 MODERADA
   Preservación geométrica: ✅ MANTENIDA
   Escalado a problemas complejos: 🟡 ACEPTABLE

🌟 RESULTADO GENERAL: ¡PROGRESO SIGNIFICATIVO!

💾 Gráficos guardados:
   - orbit_predictions_modelo_base.png
   - orbit_predictions_modelo_expandido.png
   - orbit_predictions_modelo_escalado.png
   - learning_curves_modelo_base.png
```
