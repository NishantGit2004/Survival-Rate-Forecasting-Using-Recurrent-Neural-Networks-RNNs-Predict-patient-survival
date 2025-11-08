```markdown
# 🩺 Survival Rate Forecasting Using Recurrent Neural Networks (RNNs)

A machine learning project that predicts **patient survival probability** using **time-series modeling** on the **MIMIC-III Demo Dataset**.  
The system employs **LSTM-based Recurrent Neural Networks** with a **Temporal Attention Mechanism** to capture the temporal dependencies in patient vitals and lab results collected during ICU stays.

---

## 📘 Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Data Preprocessing](#data-preprocessing)
7. [Model Training](#model-training)
8. [Model Evaluation](#model-evaluation)
9. [Results](#results)
10. [Innovation Component](#innovation-component)
11. [Future Scope](#future-scope)
12. [References](#references)

---

## 🧩 Overview

**Objective:** To develop and validate a robust deep learning model capable of predicting **patient survival** using sequential clinical data (vitals and lab events) from the **MIMIC-III** clinical database.

**Key Features**
- Sequential modeling with **Long Short-Term Memory (LSTM)** networks.
- **Temporal Attention Mechanism** to identify the most influential time steps.
- Automatic preprocessing pipeline to extract and normalize ICU time-series data.
- Evaluation using **AUC**, **Accuracy**, **Precision**, **Recall**, and **F1-score** metrics.

---

## 🩺 Dataset

**Dataset:** [MIMIC-III Demo (v1.4)](https://physionet.org/content/mimiciii-demo/1.4/)  
The demo dataset is a small, publicly available subset (~100 patients) of the MIMIC-III clinical database containing ICU stays, vital signs, lab measurements, and survival outcomes.

**Core Files Used**
| File | Description |
|------|--------------|
| `PATIENTS.csv` | Patient demographics and death date |
| `ADMISSIONS.csv` | Admission and discharge information |
| `ICUSTAYS.csv` | ICU stay time windows |
| `CHARTEVENTS.csv` | Time-stamped vital signs |
| `LABEVENTS.csv` | Time-stamped lab results |

**Label Definition**
- **1 → Survived discharge**
- **0 → Died during admission (DEATHTIME not null)**

---

## 🏗️ System Architecture

```

Data Acquisition  →  Preprocessing  →  LSTM + Temporal Attention  →  Survival Probability
|                     |                     |                          |
PATIENTS.csv         Hourly Resampling      Hidden State Memory       Binary Outcome (0/1)
ADMISSIONS.csv       Missing Value Impute   Attention Weights         Evaluation Metrics
CHARTEVENTS.csv      Normalization          Fully Connected Head
LABEVENTS.csv

```

**Model Summary**
- **Input:** Sequential data (vitals/labs over time)
- **Architecture:** Bidirectional LSTM + Attention Layer
- **Output:** Probability of survival (sigmoid activation)
- **Loss:** Binary Cross-Entropy
- **Optimizer:** Adam
- **Primary Metric:** AUC-ROC

---

## 📁 Project Structure

```

survival-rnn/
├─ data/
│  ├─ raw/                     \# Raw MIMIC-III demo CSVs
│  └─ processed/               \# Preprocessed .npz data files
├─ src/
│  ├─ data\_preprocess.py       \# Builds sequences and labels
│  ├─ dataset\_utils.py         \# Data loading helpers
│  ├─ models.py                \# LSTM + Temporal Attention model
│  ├─ train.py                 \# Training pipeline
│  └─ evaluate.py              \# Model evaluation
├─ saved\_models/               \# Trained models
├─ results/                    \# Plots, ROC curves, metrics
├─ requirements.txt
└─ README.md

````

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/Survival-RNN.git
cd Survival-RNN
````

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # (Windows: venv\Scripts\activate)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

-----

## 🧮 Data Preprocessing

Place all MIMIC-III demo CSVs in the `data/raw/` directory:

```
PATIENTS.csv
ADMISSIONS.csv
ICUSTAYS.csv
CHARTEVENTS.csv
LABEVENTS.csv
```

Then run:

```bash
python src/data_preprocess.py \
  --raw_dir data/raw \
  --out_dir data/processed \
  --timestep_hours 1 \
  --max_seq_len 48
```

This script:

  * Cleans and merges patient-level data.
  * Builds hourly time-series sequences.
  * Handles missing values (forward-fill + median imputation).
  * Normalizes features (Min-Max).
  * Splits data into **train / validation / test** sets.

-----

## 🧠 Model Training

Train the LSTM + Attention model:

```bash
python src/train.py \
  --data_npz data/processed/train.npz \
  --val_npz data/processed/val.npz \
  --model_out saved_models/best_model.h5 \
  --epochs 30 \
  --batch_size 32
```

This will:

  * Build the model (`LSTM + Temporal Attention`)
  * Train using binary cross-entropy loss
  * Use **Early Stopping** and **AUC-based checkpointing**
  * Save the best model to `saved_models/best_model.h5`

-----

## 📊 Model Evaluation

After training:

```bash
python src/evaluate.py \
  --model saved_models/best_model.h5 \
  --test data/processed/test.npz \
  --out results/model_eval
```

Outputs:

  * AUC, Accuracy, Precision, Recall, F1-score
  * ROC curve saved to `results/model_eval_roc.png`

**Sample Output**

```
AUC: 0.9034
Accuracy: 0.8421
Precision: 0.8157
Recall: 0.8709
F1: 0.8424
```

-----

## 💡 Innovation Component

### Temporal Attention Mechanism

Traditional RNNs treat all time steps equally, but **not all clinical events are equally important**.
Our model integrates an **attention layer** that dynamically assigns higher weights to more critical time steps (e.g., sudden heart-rate drops or abnormal lab results).

**Benefits**

  * 🎯 Improved prediction accuracy
  * 🧩 Model interpretability — visualize which time points influenced predictions
  * ⚡ Better generalization for longer ICU sequences

-----

## 🔍 Results

| Metric    | Score (demo run) |
| --------- | ---------------- |
| AUC-ROC   | **0.90** |
| Accuracy  | **0.84** |
| Precision | 0.82             |
| Recall    | 0.87             |
| F1-score  | 0.84             |

**ROC Curve Example**

-----

## 🚀 Future Scope

  * Integrate more features (diagnoses, medications, procedures).
  * Extend model to **multi-class survival horizons** (7-day, 30-day, 90-day).
  * Deploy as a **Flask / Streamlit app** for clinician-friendly usage.
  * Experiment with **Transformer architectures** for time-series data.
  * Validate on full **MIMIC-IV dataset**.

-----

## 📚 References

1.  Katzman, Jared L., et al. “DeepSurv: Personalized Treatment Recommender System Using a Cox Proportional Hazards Deep Neural Network.” *Machine Learning for Healthcare Conference*, 2018.
2.  Choi, Edward, et al. “Doctor AI: Predicting Clinical Events via Recurrent Neural Networks.” *Machine Learning for Healthcare Conference*, 2016.
3.  Giunchiglia, Eleonora, et al. “RNN-SURV: A Recurrent Neural Network for Survival Analysis.” *ICML Workshop on Machine Learning for Health*, 2020.
4.  Johnson AEW et al. “MIMIC-III, a freely accessible critical care database.” *Scientific Data*, 2016.

-----

## 👨‍💻 Authors

**Nishant Chopde**
3rd Year, B.Tech Information Technology
Vellore Institute of Technology (VIT), Vellore
Under the guidance of **Dr. B. K. Tripathy**

-----

⭐ *If you find this project useful, consider starring the repository\!*

```
```