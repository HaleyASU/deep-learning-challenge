# deep-learning-challenge
A neural network model that predicts whether a funding applicant is likely to succeed, helping Alphabet Soup make smarter funding decisions.

# Alphabet Soup Charity Success Predictor

## Overview

This project uses a deep learning neural network to predict whether applicants for funding from the nonprofit Alphabet Soup are likely to be successful in their ventures. The goal is to help the organization allocate resources more strategically based on historical funding data.

The analysis was performed in two stages:
1. Building and evaluating a baseline neural network model.
2. Attempting multiple optimization strategies to improve predictive performance.

---

## Data Source

The dataset includes over 34,000 records of organizations that received funding from Alphabet Soup. Key features include:

- Application type
- Classification type
- Use case for funding
- Organization type
- Income amount
- Funding amount requested
- Special considerations
- Outcome (whether the funding was used successfully)

Target variable: `IS_SUCCESSFUL`  
Input features were preprocessed using one-hot encoding, column reduction, and standard scaling.

---

## Technologies Used

- Python
- Pandas
- scikit-learn
- TensorFlow / Keras
- Google Colab

---

## Neural Network Architecture

**Final optimized model includes:**
- Input layer with one node per feature
- Three hidden layers:
  - Layer 1: 100 neurons (ReLU)
  - Layer 2: 50 neurons (ReLU)
  - Layer 3: 20 neurons (ReLU)
- Output layer: 1 neuron (Sigmoid for binary classification)

Loss function: `binary_crossentropy`  
Optimizer: `adam`  
Metric: `accuracy`

---

## Model Performance

| Model Version            | Accuracy   | Loss     |
|--------------------------|------------|----------|
| Baseline Model           | ~73.1%     | ~0.562   |
| Optimized Model (Final)  | ~72.8%     | ~0.554   |

The final baseline model achieved an accuracy of 73.08%, which was higher than the optimized version. This suggests that while deeper or more complex networks were explored, the simpler baseline performed better on unseen data.
---

## Optimization Attempts

The following strategies were tested:

1. **Increased Neurons:**  
   Adjusted the number of neurons in each hidden layer.

2. **Added Hidden Layer:**  
   Introduced a third hidden layer to deepen the network.

3. **Dropped Non-contributing Columns:**  
   Removed the `STATUS` column due to low variance.

4. **Adjusted Epochs:**  
   Trained with fewer epochs to mitigate potential overfitting.

---

## Recommendations

For future work, consider exploring:
- Tree-based models (Random Forest or XGBoost) for performance and interpretability
- Dropout layers to reduce overfitting
- Hyperparameter tuning using tools like Keras Tuner or GridSearchCV

---

## Author

Haley Armenta  
Deep Learning Challenge – Machine Learning Module  
