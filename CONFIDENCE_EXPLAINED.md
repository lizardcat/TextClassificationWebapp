# Confidence Score Explained

## 🎯 Quick Answer

**The confidence score is the probability that the model assigns to its predicted class.**

- If the model predicts "AI-Generated" with 87% probability, the confidence is **87%**
- If the model predicts "Human-Written" with 63% probability, the confidence is **63%**

The confidence tells you: **"How sure is the model about this prediction?"**

---

## 🔢 Mathematical Formula

```python
# Step 1: Get probabilities from neural network
probabilities = model.predict_proba(features)
# Returns: [P(Human), P(AI)]
# Example: [0.3521, 0.6479]

# Step 2: Get prediction (highest probability class)
prediction = model.predict(features)
# Returns: 0 (Human) or 1 (AI)
# Example: 1 (AI, because 0.6479 > 0.3521)

# Step 3: Confidence = probability of predicted class
if prediction == 1:  # AI-Generated
    confidence = probabilities[1] * 100
else:  # Human-Written
    confidence = probabilities[0] * 100

# Example: confidence = 0.6479 * 100 = 64.79%
```

**Key Insight:** The model always predicts the class with higher probability, so **confidence is always ≥ 50%**.

---

## 📊 Complete Processing Pipeline

```
User Input: "The essay text..."
    ↓
┌──────────────────────────────────────────────┐
│ PREPROCESSING                                │
├──────────────────────────────────────────────┤
│ 1. clean_text()                              │
│    - Lowercase                               │
│    - Remove punctuation/numbers              │
│    - Trim spaces                             │
│    Output: "the essay text"                  │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ FEATURE EXTRACTION                           │
├──────────────────────────────────────────────┤
│ 2. TF-IDF Vectorizer                         │
│    - Convert text to word importance scores  │
│    Output: [0.0, 0.23, 0.0, 0.45, ...]       │
│    Shape: (1, 1000) - 1000 features          │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ NORMALIZATION                                │
├──────────────────────────────────────────────┤
│ 3. StandardScaler                            │
│    - Normalize features (mean=0, std=1)      │
│    Output: [-1.2, 0.5, -0.3, 1.1, ...]       │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ DIMENSIONALITY REDUCTION                     │
├──────────────────────────────────────────────┤
│ 4. PCA                                       │
│    - Reduce to principal components          │
│    Output: [-2.10]                           │
│    Shape: (1, 1) - reduced to 1 component    │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ NEURAL NETWORK                               │
├──────────────────────────────────────────────┤
│ 5. MLPClassifier (64→32 neurons)             │
│                                              │
│    Input: 1 neuron [-2.10]                   │
│      ↓                                       │
│    Hidden Layer 1: 64 neurons                │
│      [0.2, -1.3, 0.8, ..., 0.5]              │
│      ↓ (ReLU activation)                     │
│    Hidden Layer 2: 32 neurons                │
│      [1.1, 0.4, -0.7, ..., 0.9]              │
│      ↓ (ReLU activation)                     │
│    Output Layer: 2 neurons (raw logits)      │
│      [0.63, -0.27]                           │
│      ↓ (Softmax activation)                  │
│    Probabilities: [0.6469, 0.3531]           │
│                                              │
│ Softmax Formula:                             │
│   P(class_i) = e^(logit_i) / Σ(e^(logit_j)) │
│                                              │
│   P(Human) = e^0.63 / (e^0.63 + e^-0.27)     │
│            = 1.878 / (1.878 + 0.763)         │
│            = 0.6469 = 64.69%                 │
│                                              │
│   P(AI)    = e^-0.27 / (e^0.63 + e^-0.27)    │
│            = 0.763 / 2.641                   │
│            = 0.3531 = 35.31%                 │
└──────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────┐
│ PREDICTION & CONFIDENCE                      │
├──────────────────────────────────────────────┤
│ 6. Select highest probability                │
│    max(64.69%, 35.31%) = 64.69%              │
│    → Predicted class: 0 (Human-Written)      │
│                                              │
│ 7. Confidence = probability of predicted     │
│    confidence = 64.69%                       │
└──────────────────────────────────────────────┘
    ↓
Final Output:
{
  "prediction": "Human-Written",
  "confidence": 64.69,
  "probabilities": {
    "human": 64.69,
    "ai": 35.31
  }
}
```

---

## 🧮 Why Probabilities Sum to 100%

The **softmax function** ensures all probabilities sum to 1.0 (100%):

```python
# Raw neural network outputs (logits)
logits = [0.63, -0.27]

# Apply softmax
exp_logits = [e^0.63, e^-0.27] = [1.878, 0.763]
sum_exp = 1.878 + 0.763 = 2.641

probabilities = [
    1.878 / 2.641 = 0.6469,  # 64.69%
    0.763 / 2.641 = 0.3531   # 35.31%
]

# Verify: 64.69% + 35.31% = 100% ✓
```

**Key property**: When one probability goes up, the other must go down.

---

## 📈 Example Scenarios

### Scenario 1: High Confidence AI Text

```
Input: "In accordance with the predetermined parameters, the
        aforementioned entity proceeded to execute the designated
        protocol with optimal efficiency."

Neural Network Output:
├─ P(Human) = 12.65%
└─ P(AI)    = 87.35%

Prediction: "AI-Generated" (87.35% > 12.65%)
Confidence: 87.35% ✓ HIGH CONFIDENCE
```

**Interpretation**: The model is very sure this is AI-generated text.

---

### Scenario 2: Low Confidence (Uncertain)

```
Input: "The project was completed on time and met all requirements."

Neural Network Output:
├─ P(Human) = 52.18%
└─ P(AI)    = 47.82%

Prediction: "Human-Written" (52.18% > 47.82%)
Confidence: 52.18% ⚠️ LOW CONFIDENCE
```

**Interpretation**: The model is uncertain. The text has characteristics of both AI and human writing. You should be cautious about trusting this prediction.

---

### Scenario 3: High Confidence Human Text

```
Input: "omg I can't even rn!! yesterday was literally the CRAZIEST
        day ever lol like I'm still shook 😂"

Neural Network Output:
├─ P(Human) = 94.82%
└─ P(AI)    = 5.18%

Prediction: "Human-Written" (94.82% > 5.18%)
Confidence: 94.82% ✓ HIGH CONFIDENCE
```

**Interpretation**: The model is very confident this is human-written (informal language, typos, emoji use).

---

## 🎯 Confidence Thresholds & Reliability

| Confidence | Reliability | Should You Trust It? |
|------------|-------------|---------------------|
| **90-100%** | Very High | Yes - Model is very certain |
| **80-90%** | High | Mostly - Strong signal |
| **70-80%** | Good | Probably - Reasonable confidence |
| **60-70%** | Moderate | Maybe - Model is leaning toward this |
| **50-60%** | Low | Caution - Model is uncertain |
| **<50%** | N/A | Impossible (model predicts higher class) |

**Rule of Thumb**:
- **Above 80%**: Trust the prediction
- **60-80%**: Consider the context
- **Below 60%**: The model is guessing (close to 50/50)

---

## 🔍 What Affects Confidence?

### High Confidence (90%+) When:

1. **Clear AI patterns**:
   - Formal, overly structured language
   - Perfect grammar and punctuation
   - Generic, corporate-sounding phrases
   - No typos or colloquialisms

2. **Clear Human patterns**:
   - Informal language, slang
   - Typos, grammatical errors
   - Personal anecdotes
   - Emotional language, emojis

### Low Confidence (50-60%) When:

1. **Ambiguous text**:
   - Neutral, factual statements
   - Simple sentences
   - Common phrases used by both AI and humans

2. **Mixed signals**:
   - Formal structure but personal touches
   - Perfect grammar but informal words
   - Text that could reasonably be either

3. **Short text**:
   - Not enough information for model to decide
   - Fewer distinguishing features

---

## 💡 How to Use Confidence Scores

### For School/Research:

```python
if result['confidence'] >= 80:
    conclusion = "Strong evidence this is " + result['prediction']
elif result['confidence'] >= 65:
    conclusion = "Moderate evidence this is " + result['prediction']
else:
    conclusion = "Uncertain - needs human review"
```

### Decision Making:

- **High stakes** (academic integrity): Only act on 90%+ confidence
- **Screening** (flag for review): Use 70%+ confidence
- **Exploration** (just curious): Any confidence is interesting

### Improving Low Confidence:

If you get low confidence (<65%), try:

1. **Provide more text**: Longer passages give more signal
2. **Check for edge cases**: Very short or very generic text is hard to classify
3. **Consider context**: Use confidence as one factor, not the only factor
4. **Retrain model**: Use better training data (see TRAINING_GUIDE.md)

---

## 🧪 Testing Confidence Calculation

You can verify the calculation yourself:

```python
# In your Python environment
import app

text = "Your test text here"
result = app.predict_text(text)

# Verify the math
human_prob = result['probabilities']['human']
ai_prob = result['probabilities']['ai']

# Check 1: Probabilities sum to 100%
assert abs(human_prob + ai_prob - 100.0) < 0.01, "Probabilities don't sum to 100!"

# Check 2: Confidence matches predicted class probability
if result['prediction'] == 'Human-Written':
    assert result['confidence'] == human_prob
elif result['prediction'] == 'AI-Generated':
    assert result['confidence'] == ai_prob

# Check 3: Model predicts higher probability class
max_prob = max(human_prob, ai_prob)
assert result['confidence'] == max_prob, "Confidence should be max probability!"

print("✓ All confidence checks passed!")
```

---

## 🎓 Summary

**Confidence Score = Probability of the Predicted Class**

- Calculated using softmax function on neural network outputs
- Always between 50-100% (model predicts higher probability)
- Higher confidence = more certain prediction
- Lower confidence = model is uncertain (close to 50/50)
- Use confidence to assess prediction reliability

**Visual Representation in UI:**

```
Prediction: AI-Generated
Confidence: 87.35%

Human-Written: 12.65% [███░░░░░░░]
AI-Generated:  87.35% [████████░░]
```

The progress bars show both probabilities, and the confidence score highlights how sure the model is about its choice.
