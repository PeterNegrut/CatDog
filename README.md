# CatDog Classifier 🐱🐶

This is a simple Python script that loads a trained fastai model and predicts whether an image is a **cat** or **dog**.

## How to use

1. Make sure you have Python and fastai installed
2. Run the script:

```bash
python ModelTest.py
```

3. When prompted, enter the full path to your image (e.g. `/Users/you/Downloads/Dog.jpg`)

## Files

- `ModelTest.py`: The main prediction script
- `model.pkl`: The trained fastai model (not included here, add it yourself)

## Example

```
Prediction: Dog, Confidence: 0.9987
```

## ⚠️ Version Compatibility

This model was trained using `fastai==2.7.12`. To run `app.py`, make sure to:

```bash
pip install fastai==2.7.12 torch==1.13.1 "numpy<2"
