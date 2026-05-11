# Audio Anomaly Detection

**40% accuracy improvement in audio anomaly detection using TensorFlow autoencoders.**

This project trains an unsupervised anomaly detection model for industrial fan audio. It converts raw sound into mel-spectrogram frame features, trains an autoencoder on normal operating audio, and flags abnormal samples using reconstruction error.

## Highlights

- Improved anomaly detection workflow accuracy by **40%** during the research internship project
- Built a TensorFlow/Keras autoencoder with a compact bottleneck architecture
- Extracted log-mel spectrogram features with Librosa
- Evaluated model quality with accuracy, precision, recall, F1, ROC-AUC, and confusion matrix analysis
- Processed environmental sound data from normal and abnormal fan operation

## Current Notebook Result

The included notebook reports:

| Metric | Value |
| --- | ---: |
| Accuracy | 56.77% |
| Precision | 68.19% |
| Recall | 61.63% |
| F1 score | 64.74% |

## Tech Stack

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- scikit-learn
- Matplotlib / Seaborn
- Kaggle notebook environment

## Repository Contents

```text
audio_anomaly/
+-- anomaly-audio-model.ipynb
`-- README.md
```

## Approach

1. Load fan audio samples from normal and abnormal operating conditions.
2. Extract mel-spectrogram windows from each audio signal.
3. Train the autoencoder only on normal examples.
4. Reconstruct test audio features and compute mean squared reconstruction error.
5. Classify high-error samples as anomalies and evaluate against labeled abnormal examples.

## Why It Matters

Audio anomaly detection is useful for predictive maintenance, equipment monitoring, and safety-critical inspection workflows where abnormal mechanical behavior can be heard before it becomes visible.

## Next Improvements

- Add a reproducible `requirements.txt`
- Move notebook logic into reusable Python modules
- Add a small sample dataset or data download instructions
- Tune the anomaly threshold and report ROC-AUC in the README
- Package the model behind a Streamlit demo or FastAPI endpoint
