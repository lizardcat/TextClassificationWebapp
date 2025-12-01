# AI Essay Classifier Web Application

A Flask-based web application that uses machine learning to classify text as either AI-generated or human-written. This project features a beautiful Material Design interface with custom color scheme.

## Features

- ✨ Clean, modern Material Design UI
- 🤖 ML-powered text classification (AI vs Human)
- 📊 **Confidence scores** with visual probability bars
- 📱 Responsive design for mobile and desktop
- ⚡ Fast predictions using pre-trained models
- 🎨 Custom color palette with smooth animations
- 📈 Detailed prediction breakdown (Human % vs AI %)

## Technology Stack

- **Backend**: Flask 3.0
- **ML Libraries**: scikit-learn, NumPy
- **Frontend**: HTML5, CSS3 (Material Design), Vanilla JavaScript
- **Model**: Pre-trained Neural Network with TF-IDF + PCA pipeline

## Project Structure

```
TextClassificationWebapp/
├── app.py                          # Main Flask application
├── templates/
│   └── index.html                  # Material Design UI
├── TextClassificationWebapp/
│   └── ml_assets/                  # Pre-trained ML models
│       ├── best_nn_model.pkl       # Neural network model
│       ├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
│       ├── scaler.pkl              # Feature scaler
│       └── pca.pkl                 # PCA transformer
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## ML Pipeline

The classification pipeline follows these preprocessing steps:

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove numbers and punctuation
   - Remove extra whitespace

2. **Feature Extraction**:
   - TF-IDF vectorization
   - Standard scaling
   - PCA dimensionality reduction

3. **Prediction**:
   - Neural network classification
   - Binary output: AI-Generated (1) or Human-Written (0)

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd TextClassificationWebapp
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify ML Assets

Ensure the following model files exist in `TextClassificationWebapp/ml_assets/`:
- `best_nn_model.pkl`
- `tfidf_vectorizer.pkl`
- `scaler.pkl`
- `pca.pkl`

### Step 5: Run the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

## Usage

1. Open your browser and navigate to `http://localhost:5000`
2. Paste essay text into the text area
3. Click "Classify Text"
4. View the prediction result with:
   - Classification (AI-Generated or Human-Written)
   - Confidence score (percentage)
   - Probability breakdown (Human % vs AI %)

## API Endpoint

The application also provides a REST API endpoint for programmatic access:

### POST `/api/predict`

**Request:**
```json
{
  "text": "Your essay text here..."
}
```

**Response (Success):**
```json
{
  "success": true,
  "prediction": "AI-Generated",
  "confidence": 87.35,
  "probabilities": {
    "human": 12.65,
    "ai": 87.35
  }
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Error message here"
}
```

**Example using curl:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog."}'
```

## Color Scheme

The application uses a custom Material Design color palette:

- Primary: `#957DAD` (Medium Purple)
- Secondary: `#D291BC` (Pink Purple)
- Accent 1: `#E0BBE4` (Light Purple)
- Accent 2: `#FEC8D8` (Light Pink)
- Background: `#FFDFD3` (Peach Cream)

## Development

### Running in Debug Mode

The application runs in debug mode by default for development:

```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

For production deployment, set `debug=False` and use a production WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Modifying the UI

The entire UI is contained in `templates/index.html`. All styles are inline using CSS variables for the color scheme, making it easy to customize.

### Improving & Retraining ML Models

The current model provides a baseline accuracy. For better performance:

**See the comprehensive [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for:**
- Recommended Kaggle datasets (DAIGT V2, AI vs Human Text, etc.)
- Complete retraining script with hyperparameter tuning
- Expected accuracy improvements (baseline ~65% → optimized ~92-95%)
- Advanced techniques (ensemble methods, transformer models)
- Data collection and augmentation tips

**Quick steps:**
1. Download a Kaggle dataset (e.g., DAIGT V2 - 150K+ essays)
2. Run the training script from `TRAINING_GUIDE.md`
3. New models automatically save to `TextClassificationWebapp/ml_assets/`
4. Restart Flask application to load improved models

**Current Model:**
- Type: MLPClassifier (Neural Network)
- Architecture: 64→32 neurons, ReLU activation
- Pipeline: Text → TF-IDF → StandardScaler → PCA → Prediction
- Output: Binary classification with confidence scores

## Troubleshooting

### ML Assets Not Loading

If you see the warning "ML assets failed to load":

1. Check that all `.pkl` files exist in the correct directory
2. Verify the file paths in `app.py` (line 12)
3. Ensure the files were created with compatible scikit-learn versions

### Port Already in Use

If port 5000 is already in use, change the port in `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Use different port
```

### Import Errors

If you get import errors, ensure all dependencies are installed:

```bash
pip install -r requirements.txt --upgrade
```

## License

This project is for educational purposes.

## Acknowledgments

- Built as a school learning project
- Uses Flask web framework
- Material Design principles for UI/UX
- scikit-learn for machine learning pipeline
