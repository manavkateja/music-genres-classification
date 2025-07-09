# BeatClassifier: Music Genre Classification with LSTM

A deep learning project to classify music genres using raw audio signals.  
It extracts **MFCC features** from `.wav` files and uses an **LSTM** network for classification.  
The model is deployed with a simple **Gradio web interface** for real-time predictions.

---

## Dataset

- **GTZAN Genre Classification Dataset**
- 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- 30-second `.wav` clips
- [Available on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## Model Architecture

- **Preprocessing**: Extracted MFCC (Mel Frequency Cepstral Coefficients)
- **Model**:
  - LSTM layer with 64 units
  - Dense (ReLU)
  - Dense (Softmax for 10-class output)
- **Loss**: `categorical_crossentropy`
- **Optimizer**: `Adam`

---

## Evaluation

- Accuracy: ~58% on validation set (can be improved with deeper architectures)
- Metrics used:
  - Accuracy
  - Precision, Recall, F1-score (per genre)

---

## Web App (Gradio)

Launch the app locally or in Colab:

```python
import gradio as gr
gr.Interface(fn=predict_genre, inputs=gr.Audio(type="filepath"), outputs="text").launch()

