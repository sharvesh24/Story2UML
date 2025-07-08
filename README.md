# 📘 story2uml: Generate UML Class Diagrams from User Stories using T5

**story2uml** is a deep learning-based NLP pipeline that translates natural language user stories into structured UML class diagrams in JSON format. This tool is designed to assist software analysts and developers in automating UML diagram generation during the requirements engineering phase of software development.

---

## 🧠 About the Project

This project fine-tunes the T5 (Text-to-Text Transfer Transformer) model from Hugging Face on a custom dataset consisting of user stories and their corresponding UML class diagram representations. The model learns to convert functional requirements into a structured format that can be used to visualize system design.

---

## 📁 Repository Structure

story2uml/
├── train_t5.py # Model training script
├── predict.py # Inference on user stories
├── evaluate_t5.py # Evaluation on test set
├── utils.py # Data cleaning, formatting, loading utilities
├── uml_training_data.json # Extracted training dataset
├── User_Stories.docx # Input document containing all user stories + UML
├── uml_t5_model/ # Folder for saving fine-tuned model


---

## 💼 Dataset Domains

The model was trained on user stories spanning diverse domains:

- Fitness
- E-commerce
- Education
- Task Management
- Reviews
- Others (generic user-system interactions)

---

## 🧰 Technologies Used

- Python 3.9+
- PyTorch
- Hugging Face Transformers (T5 model)
- TensorFlow (for Keras compatibility)
- JSON (for output diagram format)
- Microsoft Word (.docx) for dataset input

---

## 🏗️ Features

- Converts user stories into UML class diagram JSON.
- Supports multiple domains.
- Fine-tuned on real-world-style stories.
- CLI interface for training, prediction, and evaluation.
- Easily extendable to support other UML diagram types.

---

## 📊 Performance

- Training samples: 75  
- Testing samples: 10  
- Final training loss: ~3.26  
- Evaluation loss: ~9.25  
- Evaluation accuracy: **88%+** (structure & text match)


![Transformers](https://img.shields.io/badge/HuggingFace-T5-yellow)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Status](https://img.shields.io/badge/Accuracy-87%25-green)
![License](https://img.shields.io/badge/License-MIT-green)
