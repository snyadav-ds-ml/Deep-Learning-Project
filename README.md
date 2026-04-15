# Multimodal VQA with Answer Verification

## Overview
This project builds a multimodal system that:
1. Generates answers from image + question (Generative)
2. Verifies if the answer is correct (Discriminative)

## Setup
pip install -r requirements.txt

## Run pipeline

# Step 1: Prepare data
python src/prepare_data.py

# Step 2: Generate answers
python src/generate_answers.py

# Step 3: Train verifier
python src/train_verifier.py

# Step 4: Evaluate
python src/evaluate.py