import re
import pickle
import numpy as np
import os
from django.conf import settings
# python
from pathlib import Path
import threading
from typing import Dict, Any

_pipeline = None
_pipeline_lock = threading.Lock()

def _load_pipeline() -> None:
    global _pipeline
    if _pipeline is not None:
        return
    with _pipeline_lock:
        if _pipeline is not None:
            return
        candidate = Path(__file__).resolve().parent / "ml_pipeline.joblib"
        if candidate.exists():
            try:
                import joblib
                _pipeline = joblib.load(candidate)
                return
            except Exception:
                try:
                    import pickle
                    with open(candidate, "rb") as f:
                        _pipeline = pickle.load(f)
                        return
                except Exception:
                    _pipeline = None
        _pipeline = None

def predict(text: str) -> Dict[str, Any]:
    """
    Predict label and score for `text`.
    Returns a dict: {'label': str, 'score': float, 'backend': 'pipeline'|'fallback'|'error', ...}
    """
    _load_pipeline()
    if _pipeline is None:
        # Simple fallback heuristic classifier (safe default)
        low = text.lower() if text else ""
        positive_tokens = ("good", "great", "positive", "happy", "yes", "love")
        label = "positive" if any(tok in low for tok in positive_tokens) else "negative"
        return {"label": label, "score": 0.6, "backend": "fallback"}

    try:
        # Prefer predict_proba if available
        if hasattr(_pipeline, "predict_proba"):
            probs = _pipeline.predict_proba([text])
            import numpy as _np
            idx = int(_np.argmax(probs, axis=1)[0])
            classes = getattr(_pipeline, "classes_", None)
            label = classes[idx] if classes is not None else str(idx)
            score = float(probs[0, idx])
            return {"label": str(label), "score": score, "backend": "pipeline"}
        # Fallback to predict
        label = _pipeline.predict([text])[0]
        return {"label": str(label), "score": 1.0, "backend": "pipeline"}
    except Exception as e:
        return {"label": "error", "score": 0.0, "backend": "error", "error": str(e)}
# Determine the base directory for asset loading
# This ensures files are loaded relative to the Django project path
ASSET_DIR = os.path.join(settings.BASE_DIR, 'prediction', 'ml_assets')

# Initialize ML components globally for efficiency
tfidf_vectorizer, scaler, pca, model = None, None, None, None


def load_ml_assets():
    """Loads all pre-trained ML components into memory."""
    global tfidf_vectorizer, scaler, pca, model

    # Only load if not already loaded (for first run)
    if model is not None:
        return

    try:
        # Construct full paths for each asset
        tfidf_path = os.path.join(ASSET_DIR, 'tfidf_vectorizer.pkl')
        scaler_path = os.path.join(ASSET_DIR, 'scaler.pkl')
        pca_path = os.path.join(ASSET_DIR, 'pca.pkl')
        model_path = os.path.join(ASSET_DIR, 'classification_model.pkl')

        # Load the fitted assets
        with open(tfidf_path, 'rb') as f:
            tfidf_vectorizer = pickle.load(f)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print("--- All ML assets loaded successfully. ---")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find required ML asset file: {e}. Check your 'prediction/ml_assets' directory.")
    except Exception as e:
        print(f"An unexpected error occurred during asset loading: {e}")


# Call the loading function when the Django app starts
load_ml_assets()


def clean_text(text):
    """
    Applies the specified text cleaning steps.
    """
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def get_prediction(raw_text: str) -> str:
    """
    Executes the full preprocessing and prediction pipeline on a single text input.
    """
    # Check if model components are available
    if not all([model, tfidf_vectorizer, scaler, pca]):
        # This occurs if load_ml_assets failed
        return "ERROR: Model components are not loaded. Cannot predict."

    try:
        # Step 1: Clean the input text
        x_test_cleaned = clean_text(raw_text)

        # Step 2: TF-IDF Transformation (must use .transform and expects an iterable)
        x_test_features = tfidf_vectorizer.transform([x_test_cleaned]).toarray()

        # Step 3: Scaling (must use .transform)
        x_test_scaled = scaler.transform(x_test_features)

        # Step 4: PCA (must use .transform)
        x_test_pca = pca.transform(x_test_scaled)

        # Step 5: Prediction
        prediction = model.predict(x_test_pca)[0]

        # Convert numeric prediction to a human-readable label
        if prediction == 1:
            return "AI-Generated"
        else:
            return "Human-Written"

    except Exception as e:
        print(f"Prediction Error: {e}")
        return "ERROR: An internal processing error occurred."


# Example: If you run this file directly for testing
if __name__ == '__main__':
    sample_text = "The quick brown fox jumps over the lazy dog, showcasing linguistic complexity."
    result = get_prediction(sample_text)
    print(f"\nTest Input: '{sample_text[:50]}...'")
    print(f"Test Result: {result}")