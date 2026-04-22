# 🧠 Visual Question Answering with Answer Verification

## 📌 Overview

This project builds a **Visual Question Answering (VQA)** system that not only answers questions about an image but also **checks if the answer is correct**.

👉 Instead of trusting the model blindly, we add a **Verifier model** to improve reliability.

---

## 🚀 What This Project Does

1. Takes an **image + question**
2. Generates an answer using a pretrained model (**BLIP**)
3. Verifies if the answer is **correct or not**
4. Shows the final result in a simple **Streamlit app**

---

## 🏗️ Architecture

Image + Question → BLIP Model → Generated Answer → Verifier → ✅/❌

---

## 🧩 Components Explained

### 1. Data Preparation (`prepare_data.py`)
- Loads VQA dataset
- Cleans and formats data
- Splits into train and validation

### 2. Answer Generation (`generate_answers.py`)
- Uses BLIP (pretrained model)
- Input: image + question  
- Output: predicted answer

### 3. Verifier Data Creation (`utils.py`)
Creates:
- Correct answer → label = 1  
- Wrong answer → label = 0  

### 4. Verifier Model (`models.py`)
- Text Encoder (BiLSTM)
- Image features (BLIP)
- Fusion: [image, text, image * text]
- Classifier (MLP)

### 5. Training (`train_verifier.py`)
- Binary classification using BCE Loss

### 6. Evaluation (`evaluate.py`)
- Generator: Exact Match, F1
- Verifier: Accuracy, Precision, Recall, F1

### 7. Demo App (`app.py`)
- Built with Streamlit
- Upload image + ask question
- Shows answer + correctness

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- BLIP
- Streamlit

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python src/prepare_data.py
python src/generate_answers.py
python src/train_verifier.py
python src/evaluate.py
streamlit run app.py
```

---

## 💡 Key Idea

👉 Generate answer  
👉 Then verify it  

This improves reliability of VQA systems.

---

## 🙌 Conclusion

This project demonstrates multimodal AI with answer verification.

⭐ Star the repo if you like it!
