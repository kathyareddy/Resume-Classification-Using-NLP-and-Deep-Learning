# Resume Classification Using NLP and Deep Learning

An AI-powered system that classifies resumes into predefined job categories using Natural Language Processing (NLP) and a Deep Learning model. It helps recruiters or automated systems filter resumes based on relevant skills.

---

## Overview

This project takes a dataset of resumes labeled by job domain (e.g., Data Science, HR, Design, etc.), cleans and processes the text using **NLP techniques**, then builds a **deep learning classification model** to predict the most suitable category for any given resume.

---

## Tech Stack

- **Python**
- **Pandas, NumPy** – Data Handling
- **SpaCy, NLTK** – NLP Preprocessing
- **TF-IDF** – Feature Extraction
- **TensorFlow / Keras** – Deep Learning
- **Matplotlib, Seaborn** – Visualizations

---

## Dataset

- **Source**: [AI_Resume_Screening.csv](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset)
- **Attributes**: `Category`, `Resume` (Text)
- **Categories Include**: Data Science, HR, Design, Testing, Operations, Sales, etc.

---

## File Structure

```
Resume-Classification-Using-NLP-and-Deep-Learning/
│
├── resume_classification_nlp_dl.ipynb   # Jupyter notebook with full workflow
├── AI_Resume_Screening.csv              # Dataset used for training
└── README.md                            # Project documentation
```

---

## Key Steps

1. **Text Cleaning & Preprocessing**  
   - Lowercasing, removing punctuation/special characters  
   - Stopword removal, tokenization, lemmatization using SpaCy  

2. **Feature Extraction**  
   - TF-IDF vectorization of resume text  

3. **Model Building**  
   - Sequential Deep Neural Network using TensorFlow/Keras  
   - Layers: Dense + Dropout  
   - Output Layer: Softmax over job categories  

4. **Evaluation**  
   - Accuracy, Confusion Matrix, Classification Report  

---

## Results

- **Validation Accuracy**: ~87%  
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Epochs**: 20+


---

## Future Enhancements

- Convert model into a web API or Streamlit app for HR use  
- Integrate resume upload and parsing (PDF to text)  
- Experiment with LSTM or BERT-based embeddings for higher accuracy  
- Add named entity recognition to extract key skills  

---


