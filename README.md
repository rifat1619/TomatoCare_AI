# 🍅 TomatoCare AI — Streamlit

AI-powered tomato leaf disease diagnosis using the trained DenseNet121/Keras model.

## Repository structure

```text
TomatoCare_AI/
├── app.py
├── model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

## Important

`model.keras` is the trained model file. It is stored in GitHub using Git LFS because it is larger than the normal GitHub web-upload limit.

The app follows the uploaded training notebook's inference preprocessing:

1. RGB image
2. Resize to `224 × 224`
3. CLAHE with `clipLimit=2.0` and `tileGridSize=(8, 8)`
4. Convert to `float32`
5. Divide by `255`

The model is loaded with `compile=False`, so training-only loss/optimizer configuration is not required during inference.

## Deploy on Streamlit Community Cloud

1. Push these files to your GitHub repository.
2. Keep `model.keras` in the repository root.
3. Open Streamlit Community Cloud.
4. Choose **Create app**.
5. Repository: `rifat1619/TomatoCare_AI`
6. Branch: `main`
7. Main file: `app.py`
8. Deploy.

## Model classes

The training notebook creates class indices from the dataframe generator. The deployed app uses this 9-class order:

- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Miner
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Verticillium Wilt
- Healthy Leaf

> Academic/research prototype. Predictions should be validated by an appropriate agricultural expert before treatment or crop-management decisions.
