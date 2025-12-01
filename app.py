from flask import Flask, render_template, request, jsonify
import joblib
import re
import os
from pathlib import Path
import warnings

# Suppress scikit-learn version warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Determine the base directory for ML assets
BASE_DIR = Path(__file__).resolve().parent
ML_ASSETS_DIR = BASE_DIR / 'TextClassificationWebapp' / 'ml_assets'

# Global variables for ML components
tfidf_vectorizer = None
scaler = None
pca = None
model = None


def load_ml_assets():
    """Load all pre-trained ML components into memory."""
    global tfidf_vectorizer, scaler, pca, model

    try:
        # Load the fitted assets using joblib
        tfidf_vectorizer = joblib.load(ML_ASSETS_DIR / 'tfidf_vectorizer.pkl')
        scaler = joblib.load(ML_ASSETS_DIR / 'scaler.pkl')
        pca = joblib.load(ML_ASSETS_DIR / 'pca.pkl')

        # Try to load the model with error handling for numpy compatibility
        try:
            model = joblib.load(ML_ASSETS_DIR / 'best_nn_model.pkl')
        except (ValueError, AttributeError) as model_error:
            # If there's a numpy random state issue, try with allow_pickle=False
            print(f"⚠ Warning loading model: {model_error}")
            print("  Attempting alternative loading method...")
            import pickle
            with open(ML_ASSETS_DIR / 'best_nn_model.pkl', 'rb') as f:
                model = pickle.load(f)

        print("✓ All ML assets loaded successfully")
        print(f"  - TF-IDF Vectorizer: {type(tfidf_vectorizer).__name__}")
        print(f"  - Scaler: {type(scaler).__name__}")
        print(f"  - PCA: {type(pca).__name__}")
        print(f"  - Model: {type(model).__name__}")
        return True

    except FileNotFoundError as e:
        print(f"✗ ERROR: Could not find ML asset file: {e}")
        print(f"  Looking in: {ML_ASSETS_DIR}")
        return False
    except Exception as e:
        print(f"✗ An unexpected error occurred during asset loading: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def clean_text(text):
    """
    Clean text using the specified preprocessing steps:
    - Convert to lowercase
    - Remove numbers and punctuation
    - Remove extra spaces
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Lowercase
    text = text.lower()

    # Remove numbers and punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def predict_text(raw_text):
    """
    Execute the full preprocessing and prediction pipeline.

    Pipeline steps:
    1. Clean text (lowercase, remove punctuation/numbers, remove extra spaces)
    2. TF-IDF transformation
    3. Scaling
    4. PCA transformation
    5. Model prediction

    Returns:
        dict: Contains 'success', 'prediction', and optional 'error' keys
    """
    # Validate that models are loaded
    if not all([model, tfidf_vectorizer, scaler, pca]):
        return {
            'success': False,
            'error': 'ML models are not loaded. Please restart the application.'
        }

    # Validate input
    if not raw_text or not raw_text.strip():
        return {
            'success': False,
            'error': 'Please enter some text to classify.'
        }

    try:
        # Step 1: Clean the input text
        x_test_cleaned = clean_text(raw_text)

        # Check if text is empty after cleaning
        if not x_test_cleaned:
            return {
                'success': False,
                'error': 'Text contains no valid characters after preprocessing.'
            }

        # Step 2: TF-IDF Transformation
        x_test_features = tfidf_vectorizer.transform([x_test_cleaned]).toarray()

        # Step 3: Scaling
        x_test_scaled = scaler.transform(x_test_features)

        # Step 4: PCA
        x_test_pca = pca.transform(x_test_scaled)

        # Step 5: Prediction
        prediction = model.predict(x_test_pca)[0]

        # Convert numeric prediction to human-readable label
        if prediction == 1:
            result = "AI-Generated"
        else:
            result = "Human-Written"

        return {
            'success': True,
            'prediction': result
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        return {
            'success': False,
            'error': 'An error occurred during text classification. Please try again.'
        }


@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page route that handles both displaying the form and processing predictions."""
    if request.method == 'POST':
        text = request.form.get('text', '')
        result = predict_text(text)

        return render_template('index.html',
                             text=text,
                             result=result)

    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions (optional, for programmatic access)."""
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({
            'success': False,
            'error': 'No text provided'
        }), 400

    result = predict_text(data['text'])
    return jsonify(result)


# Load ML assets when the app starts
with app.app_context():
    assets_loaded = load_ml_assets()
    if not assets_loaded:
        print("\n⚠ WARNING: App started but ML assets failed to load!")
        print("  The application will not work correctly.\n")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
