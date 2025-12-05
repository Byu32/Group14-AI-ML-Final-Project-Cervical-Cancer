# Group14-AI-ML-Final-Project-Cervical-Cancer

## Project Overview
This project implements a computer vision pipeline for the automated classification of cervical cancer from cervical images. The system classifies images into three categories: **Normal**, **Precancerous**, and **Cancerous**.

---

## Installation & Setup

This project is optimized for Google Colab. Please run the following steps in a Colab cell to set up the environment.

### Step 1: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: Clone Repository
```python
!git clone https://github.com/Byu32/Group14-AI-ML-Final-Project-Cervical-Cancer.git
%cd Group14-AI-ML-Final-Project-Cervical-Cancer
```

### Step 3: Install Dependencies
```python
!pip install -r requirements.txt
```

---

## How to Run

### 1. Data Preparation
*   Run **`fetch_jphiego.ipynb`** to download the Jhpiego flashcards PDF, automatically extract cervical images, and parse diagnostic labels into a CSV dataset.
*   Run **`combine_dataset.ipynb`** to consolidate the extracted Jhpiego images and the IARC dataset into a unified, structured image bank with standardized labels.

### 2. Preprocessing
*   Run **`Clean_Dataset.ipynb`** for data cleaning.
*   Run **`preprocessing.ipynb`** for splitting, enhancement (CLAHE), and augmentation.

### 3. Model Training
*   Run **`training_F1.ipynb`** to fine-tune the EfficientNet-B0 model using transfer learning, optimizing for F1-Score to handle class imbalance, and save the best-performing model weights.
*   Run **`train_AUC.ipynb`** to train the EfficientNet model optimizing for AUC score.

### 4. Model Demonstration
*   **Activate Backend**: Navigate to the **`_demo/`** folder in your terminal and run `python app.py`.
*   **Launch Interface**: Open **`GUI.html`** in a web browser.
*   **Load Model**: Click "Admin Model Upload", enter the security code **"cerviGOAT"**, and upload the **`final_model_full.pkl`** file.
*   **Inference**: Upload sample images from the **`via_dataset`** folder to receive real-time diagnoses (Normal/Precancerous/Cancerous) with confidence levels.
