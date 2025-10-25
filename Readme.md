## 🧠 AI & Machine Learning Practical Tasks
# 📘 Overview

This project demonstrates hands-on applications of Machine Learning (ML), Deep Learning (DL), 
and Natural Language Processing (NLP) using Scikit-learn, TensorFlow, and spaCy.
Each task explores a key area of AI — from classical algorithms to deep neural networks and ethical considerations.

## Links
📄 [View the full report (PDF)]   https://drive.google.com/file/d/1TXUU36VT5A758YCBpCwEEut6zYG-3FVD/view?usp=sharing
    [Streamlit Live Demo Link]     http://localhost:8501/

# 🚀 Tasks Breakdown
# 🧩 Task 1: Classical ML with Scikit-learn

Dataset: Iris Species
Goal: Train a Decision Tree Classifier to predict iris flower species.
# Key Steps:

* Preprocessed the dataset (handled missing values, label encoding).
* Trained a DecisionTreeClassifier.
* Evaluated using accuracy, precision, and recall.
* Visualized results via a confusion matrix.

📁 File: Iris_Decision_Tree.py
📊 Tech: Python, Pandas, Scikit-learn, Matplotlib

# 🧠 Task 2: Deep Learning with TensorFlow

Dataset: MNIST Handwritten Digits
Goal: Build a Convolutional Neural Network (CNN) to classify digits (0–9).
# Achievements:
* Preprocessed image data and normalized pixel values.
* Built and trained a CNN using Conv2D, MaxPooling2D, and Dense layers.
* Reached >97% test accuracy.
* Visualized predictions on 5 sample test images.

📁 File: mnist_cnn_classifier.py
⚙️ Tech: TensorFlow, Keras, NumPy, Matplotlib

# 💬 Task 3: NLP with spaCy

Dataset: Sample Amazon Product Reviews
Goal: Extract named entities (brands/products) and perform sentiment analysis.
# Key Features:

* Used spaCy’s Named Entity Recognition (NER) for brand detection.
* Applied a rule-based sentiment analyzer for positive/negative classification.
* Displayed results with extracted entities and sentiment labels.

📁 File: nlp_spacy_sentiment.py
🧰 Tech: spaCy, Python

# ⚖️ Part 3: Ethics & Optimization

- Bias in Models:

* MNIST: Style and cultural bias due to uniform handwriting samples.
* Amazon Reviews: Linguistic bias in sentiment detection.

- Mitigation:

* TensorFlow Fairness Indicators to visualize fairness across subgroups.
* spaCy rule-based tuning to handle context and reduce bias.

- Debugging Challenge: Fixed TensorFlow errors like input dimension mismatches and incorrect loss functions.

📄 File: ethics_and_debugging_notes.txt


# 🧩 Summary of Tools Used
Category	Tools/Libraries	Purpose
Machine Learning	Scikit-learn, Pandas	Classical ML & Decision Trees
Deep Learning	TensorFlow / Keras	CNN Model Training
NLP	spaCy	Entity Recognition & Sentiment
Visualization	Matplotlib	Graphs & Confusion Matrix
Ethics	TensorFlow Fairness Indicators, Custom Rules	Bias detection & mitigation

# 🧾 Ethics Report (Summary)

Ethical considerations are vital to ensure fairness and transparency in AI systems. 
The MNIST CNN model may perform unevenly on diverse handwriting styles, while the spaCy-based sentiment 
analyzer might misinterpret sarcasm or dialects. Tools like TensorFlow Fairness Indicators help detect subgroup performance gaps, 
and spaCy’s rule-based refinement supports contextual fairness. Together, these techniques foster responsible and inclusive AI development.

> 💡 How to Run
# Create and activate environment (optional but recommended)
python -m venv ai_env
ai_env\Scripts\activate   # (Windows)
# or
source ai_env/bin/activate  # (Mac/Linux)
# Install dependencies
pip install -r requirements.txt


> Run Each Task:
# Task 1: Classical ML
python Iris_DecisionTree_Notebook.py
# Task 2: Deep Learning (CNN)
python mnist_cnn_classifier.py
# Task 3: NLP
python nlp_spacy_sentiment.py


# 🧰 Requirements
tensorflow
scikit-learn
pandas
numpy
matplotlib
spacy
(Install the spaCy model: python -m spacy download en_core_web_sm)

## 🏁 Conclusion

This project highlights the core stages of modern AI workflows — from data preprocessing 
and model training to evaluation and ethical analysis.
By applying classical ML, CNN-based deep learning, and NLP techniques, the project provides 
a full view of how intelligent systems can be built, tested, and improved responsibly.