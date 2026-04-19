# Multimodal VQA with Answer Verification

This final version is designed to satisfy the computer vision project rubric more cleanly.

## Project idea
This system uses both image data and sequence data:
- **Image modality**: image input from VQAv2
- **Sequence modality**: natural-language questions and generated answers

It contains both:
- **Generative component**: BLIP generates an answer from image + question
- **Discriminative component**: a verifier predicts whether a candidate answer is correct for the image-question pair

## Architecture
1. `prepare_data.py`
   - loads `merve/vqav2-small`
   - creates train / validation / test splits
2. `generate_answers.py`
   - runs BLIP VQA on each split
   - saves generated answers
3. `train_verifier.py`
   - builds positive and negative answer candidates
   - uses BLIP pooled embeddings for image and text
   - trains a binary MLP verifier
4. `evaluate.py`
   - evaluates generator exact-match and token-F1
   - evaluates verifier accuracy / precision / recall / F1
   - evaluates end-to-end selective accuracy after verifier filtering
5. `app.py`
   - Streamlit demo for presentation

## Folder structure
```text
final_vqa_project/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── processed/
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── predictions/
└── src/
    ├── evaluate.py
    ├── generate_answers.py
    ├── models.py
    ├── prepare_data.py
    ├── train_verifier.py
    └── utils.py
```

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python src/prepare_data.py
python src/generate_answers.py
python src/train_verifier.py
python src/evaluate.py --split test
streamlit run app.py
```

## Output artifacts for report/presentation
- `data/processed/metadata.json`
- `outputs/predictions/*_with_pred.pkl`
- `outputs/checkpoints/verifier.pt`
- `outputs/metrics/verifier_training_history.json`
- `outputs/metrics/evaluation_test.json`

## What changed from the original repo
- added a **test split**
- fixed fragile BLIP embedding handling
- made verifier embedding dimension dynamic
- added real **generator + verifier + end-to-end** evaluation
- added metrics JSON outputs for the report
- added missing dependency for Streamlit
- cleaned pipeline for presentation/demo use
