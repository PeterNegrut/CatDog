from fastai.vision.all import *
import gradio as gr

learn = load_learner("model (1).pkl")  # Rename if your file is model (1).pkl

def classify_image(img):
    pred_class, pred_idx, probs = learn.predict(img)
    label = "Dog" if pred_class == "False" else "Cat"
    return f"{label} ({probs[pred_idx]:.4f})"

gr.Interface(fn=classify_image,
             inputs=gr.Image(type="pil"),
             outputs="text",
             title="Cat vs Dog Classifier 🐶🐱").launch()
