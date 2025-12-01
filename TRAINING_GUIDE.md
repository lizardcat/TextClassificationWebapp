# Model Training & Improvement Guide

This guide shows you how to retrain and improve your AI text classifier using Kaggle datasets.

## 📊 Recommended Kaggle Datasets

### 1. **DAIGT V2 Train Dataset** (Recommended)
- **URL**: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
- **Size**: ~600MB, 150K+ essays
- **Labels**: AI-generated (ChatGPT, GPT-3.5, GPT-4, Claude) vs Human-written
- **Quality**: High-quality, recent data (2023-2024)
- **Best for**: General essay classification

### 2. **AI vs Human Text Detection**
- **URL**: https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text
- **Size**: ~50MB, 50K+ samples
- **Labels**: Binary (AI/Human)
- **Quality**: Good variety of writing styles
- **Best for**: Smaller, faster training

### 3. **GPT-3 Generated Text Detection**
- **URL**: https://www.kaggle.com/datasets/nbroad/gpt-3-generated-text
- **Size**: ~20MB, 20K+ samples
- **Labels**: Binary (GPT-3/Human)
- **Best for**: Baseline model training

### 4. **Essay Scoring Dataset** (for additional features)
- **URL**: https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2
- **Use**: Combine with AI detection for multi-task learning

## 🔧 How to Retrain Your Model

### Step 1: Download Dataset

```bash
# Install Kaggle API
pip install kaggle

# Configure API key (get from https://www.kaggle.com/settings)
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d thedrcat/daigt-v2-train-dataset
unzip daigt-v2-train-dataset.zip
```

### Step 2: Create Training Script

Create `train_model.py`:

```python
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ============================================
# 1. LOAD DATA
# ============================================
print("Loading data...")
# Adjust path based on your downloaded dataset
df = pd.read_csv('train_essays.csv')  # or your dataset file

# Assume columns: 'text' and 'label' (0=Human, 1=AI)
# If your dataset has different column names, adjust:
# df = df.rename(columns={'essay': 'text', 'generated': 'label'})

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")


# ============================================
# 2. TEXT PREPROCESSING
# ============================================
def clean_text(text):
    """
    Clean text - MUST match your webapp's preprocessing!
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove numbers and punctuation (keep only letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("Cleaning text...")
df['cleaned_text'] = df['text'].apply(clean_text)

# Remove empty texts
df = df[df['cleaned_text'].str.len() > 0]
print(f"After cleaning: {df.shape}")


# ============================================
# 3. TRAIN/TEST SPLIT
# ============================================
X = df['cleaned_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# ============================================
# 4. FEATURE EXTRACTION - TF-IDF
# ============================================
print("\nExtracting TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,      # Increase for better accuracy (up to 10000)
    ngram_range=(1, 3),     # Use 1-3 word combinations
    min_df=5,               # Ignore rare words
    max_df=0.8              # Ignore very common words
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

print(f"TF-IDF shape: {X_train_tfidf.shape}")


# ============================================
# 5. SCALING
# ============================================
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_tfidf)
X_test_scaled = scaler.transform(X_test_tfidf)


# ============================================
# 6. DIMENSIONALITY REDUCTION - PCA
# ============================================
print("Applying PCA...")
pca = PCA(n_components=200, random_state=42)  # Adjust components (100-500)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA shape: {X_train_pca.shape}")
print(f"Explained variance: {pca.explained_variance_ratio_.sum():.2%}")


# ============================================
# 7. TRAIN NEURAL NETWORK
# ============================================
print("\nTraining Neural Network...")
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # Increase layers for better accuracy
    activation='relu',
    solver='adam',
    max_iter=300,                      # Increase iterations
    random_state=42,
    early_stopping=True,               # Stop if not improving
    validation_fraction=0.1,
    verbose=True,
    learning_rate_init=0.001
)

model.fit(X_train_pca, y_train)


# ============================================
# 8. EVALUATE MODEL
# ============================================
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)

# Training accuracy
train_pred = model.predict(X_train_pca)
train_acc = accuracy_score(y_train, train_pred)
print(f"\nTraining Accuracy: {train_acc:.2%}")

# Test accuracy
y_pred = model.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.2%}")

# Detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Probability scores
y_pred_proba = model.predict_proba(X_test_pca)
print(f"\nAverage confidence: {y_pred_proba.max(axis=1).mean():.2%}")


# ============================================
# 9. SAVE MODELS
# ============================================
print("\nSaving models...")
output_dir = 'TextClassificationWebapp/ml_assets'

joblib.dump(tfidf_vectorizer, f'{output_dir}/tfidf_vectorizer.pkl')
joblib.dump(scaler, f'{output_dir}/scaler.pkl')
joblib.dump(pca, f'{output_dir}/pca.pkl')
joblib.dump(model, f'{output_dir}/best_nn_model.pkl')

print("✓ All models saved successfully!")
print(f"\nFinal Test Accuracy: {test_acc:.2%}")
```

### Step 3: Run Training

```bash
python train_model.py
```

Expected output:
```
Loading data...
Dataset shape: (150000, 2)
...
Final Test Accuracy: 92.5%
✓ All models saved successfully!
```

## 🚀 Hyperparameter Tuning for Better Accuracy

### TF-IDF Parameters
```python
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,     # ↑ More features = better but slower
    ngram_range=(1, 3),     # Use 1-3 word phrases
    min_df=5,               # Minimum document frequency
    max_df=0.8,             # Maximum document frequency
    sublinear_tf=True       # Use log scaling
)
```

### PCA Parameters
```python
pca = PCA(
    n_components=300,       # ↑ More components = retain more info
    random_state=42
)
```

### Neural Network Parameters
```python
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),  # ↑ Larger network
    activation='relu',                   # or 'tanh'
    solver='adam',                       # Best for most cases
    alpha=0.0001,                        # L2 regularization
    batch_size='auto',                   # or 200, 500
    learning_rate='adaptive',            # Adjust learning rate
    max_iter=500,                        # ↑ More iterations
    early_stopping=True,
    validation_fraction=0.1
)
```

## 📈 Expected Improvements

| Model Version | Accuracy | Notes |
|---------------|----------|-------|
| Current (baseline) | ~65-75% | Small training set |
| DAIGT V2 (basic) | ~85-90% | Larger, better data |
| DAIGT V2 (tuned) | ~92-95% | Optimized hyperparameters |
| Ensemble models | ~96-98% | Multiple models combined |

## 🔬 Advanced Improvements

### 1. **Use Ensemble Methods**
Combine multiple models for better accuracy:

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Create multiple models
mlp = MLPClassifier(hidden_layer_sizes=(128, 64))
lr = LogisticRegression(max_iter=1000)
svm = SVC(kernel='rbf', probability=True)

# Combine them
ensemble = VotingClassifier(
    estimators=[('mlp', mlp), ('lr', lr), ('svm', svm)],
    voting='soft'  # Use probability averaging
)

ensemble.fit(X_train_pca, y_train)
```

### 2. **Add More Features**
Beyond TF-IDF, add linguistic features:

```python
def extract_features(text):
    """Extract additional features"""
    return {
        'avg_word_length': np.mean([len(w) for w in text.split()]),
        'sentence_count': text.count('.'),
        'unique_words': len(set(text.split())) / len(text.split()),
        'punctuation_ratio': sum(c in '.,!?' for c in text) / len(text),
        # Add more features...
    }
```

### 3. **Use Transformer Models** (Advanced)
For state-of-the-art accuracy (95-99%):

```python
# Use pre-trained BERT or RoBERTa
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

# Fine-tune on your dataset
# (requires more code and GPU for training)
```

## 📝 Data Collection Tips

### Creating Your Own Dataset

1. **Human-written texts**:
   - Reddit posts
   - News articles
   - Student essays (with permission)
   - Wikipedia articles
   - Academic papers

2. **AI-generated texts**:
   - ChatGPT responses
   - Claude responses
   - GPT-4 outputs
   - Other LLM outputs

3. **Balance**:
   - Keep 50/50 ratio of AI vs Human
   - Ensure diverse writing styles
   - Match text length distributions

### Data Augmentation

```python
# Generate variations of existing texts
import nlpaug.augmenter.word as naw

aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(original_text)
```

## 🎯 Testing Your New Model

After retraining, test with sample texts:

```python
# Test samples
test_samples = [
    "AI-like: In accordance with the aforementioned directives...",
    "Human-like: I can't believe what happened yesterday! So crazy..."
]

for text in test_samples:
    result = predict_text(text)
    print(f"Text: {text[:50]}")
    print(f"Prediction: {result['prediction']} ({result['confidence']}%)\n")
```

## 📚 Additional Resources

- **Kaggle Learn**: https://www.kaggle.com/learn/intro-to-machine-learning
- **Scikit-learn Docs**: https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- **Text Classification Guide**: https://developers.google.com/machine-learning/guides/text-classification

## ⚠️ Important Notes

1. **Preprocessing Consistency**: Your training preprocessing MUST match the webapp's `clean_text()` function exactly
2. **Version Compatibility**: Use matching scikit-learn versions (>=1.7.2)
3. **Save with joblib**: Always use `joblib.dump()` not `pickle.dump()`
4. **Test thoroughly**: Validate on diverse, unseen texts before deploying

## 🎓 For Your School Project

To demonstrate learning and improvement:

1. **Baseline**: Document current model accuracy (~65%)
2. **Retrain**: Use DAIGT V2 dataset, show improvement (~85-90%)
3. **Tune**: Optimize hyperparameters, show further improvement (~92-95%)
4. **Compare**: Create a table showing before/after metrics
5. **Visualize**: Plot confusion matrix, ROC curves, confidence distributions

This shows systematic scientific approach to ML improvement!
