# 🎙️ Deepfake Voice Detection System
### Pakistani Urdu-English Code-Switched Audio Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-95.4%25-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

This project is a **deepfake voice detection system** specifically designed for **Pakistani code-switched speech** — audio where speakers naturally mix Urdu and English within the same sentence (e.g., *"Maine yeh project complete kar liya hai finally"*).

Most existing deepfake detection systems are trained on English-only or single-language audio. This project addresses a genuine research gap by building a system that works on **multilingual code-switched speech** — the natural speaking style of most urban Pakistanis.

The system can:
- Analyze any audio file and determine whether it is a **real human voice** or **AI generated (deepfake)**
- Provide a **confidence score** for the prediction
- Show a **timeline** of which seconds in the audio are suspicious
- Explain **why** the audio is considered fake using SHAP feature importance
- Generate a **professional PDF report** with all findings
- Run as a **web application** where users can upload audio and get instant results

**Developed as part of the NAVTTC AI/ML program at NETSOL Institute of Artificial Intelligence (NIAI), Lahore, Pakistan.**

---

## 🏆 Key Results

| Model | Accuracy | Training Time |
|-------|----------|---------------|
| Random Forest | **95.4%** ✅ Selected | 0.16 seconds |
| SVM | 98.8% | 0.04 seconds |
| Gradient Boosting | 92.5% | 0.84 seconds |

> Random Forest was selected despite SVM's higher accuracy because **SHAP TreeExplainer** — used for generating explainability reports — is only compatible with tree-based models. This is a deliberate engineering tradeoff: explainability was prioritized over marginal accuracy gain.

---

## 🗂️ Project Structure

```
deepfake-voice-detection/
│
├── 01_data_preparation.ipynb     # Step 1: Dataset setup and chunking
├── 02_model_training.ipynb       # Step 2: Feature extraction and model training
├── 03_report_generation.ipynb    # Step 3: Prediction and PDF report generation
├── 04_live_detection.ipynb       # Step 4: Real-time microphone detection
├── app.py                        # Streamlit web application
├── requirements.txt              # All required Python libraries
├── README.md                     # This file
│
├── models/                       # Saved trained model files
│   ├── model.pkl                 # Trained Random Forest classifier
│   ├── scaler.pkl                # StandardScaler fitted on training data
│   ├── real_means.npy            # Mean MFCC values of real voices
│   ├── real_stds.npy             # Std deviation of real voice MFCC values
│   └── model_info.txt            # Model accuracy and selection details
│
├── dataset/                      # Audio dataset (not included in repo — see below)
│   ├── real/                     # Original real human voice samples (151 files)
│   ├── fake/                     # Original AI generated voice samples (151 files)
│   ├── real_chunks/              # 3-second chunks from real files (536 chunks)
│   └── fake_chunks/              # 3-second chunks from fake files (327 chunks)
│
├── audio_reports/                # Generated PDF reports
│   └── charts/                   # Chart images used in PDF reports
│
└── metadata.csv                  # Dataset labels and metadata
```

---

## 🛠️ Section 1 — Tools and Technologies

Every tool in this project was chosen for a specific reason. Here is a complete explanation of each:

### Python 3.10+
The core programming language. Python is the industry standard for machine learning and data science due to its rich ecosystem of libraries and readable syntax.

### Jupyter Notebook
Used for the 4 main pipeline notebooks. Jupyter allows running code cell by cell, which is ideal for data science workflows where you need to inspect results at each step before proceeding.

### librosa
A Python library specifically designed for audio analysis. Used to:
- Load audio files at a standardized sample rate (16kHz)
- Extract MFCC (Mel Frequency Cepstral Coefficient) features from audio
- Calculate RMS energy for spike detection during cleaning
- All audio processing in this project relies on librosa

### noisereduce
Used in the audio cleaning pipeline to reduce background noise. It works by estimating the noise profile from the first 0.3 seconds of audio and then suppressing that noise pattern throughout the clip. This ensures that model predictions are not influenced by recording environment noise.

### scipy
Used specifically for `gaussian_filter1d` — a smoothing function applied to the audio gain mask during spike removal. Without smoothing, abrupt gain changes create audible clicks in the cleaned audio.

### soundfile
Used to write processed audio chunks to disk as WAV files. Works alongside librosa for all audio read/write operations.

### NumPy
Fundamental library for numerical operations. Used throughout for:
- Array manipulations on audio data
- MFCC feature matrix operations
- Statistical calculations (mean, standard deviation) for normal range charts

### Pandas
Used for reading and writing the metadata CSV file that tracks all dataset samples with their labels, sources, and dates.

### scikit-learn
The core machine learning library. Used for:
- `RandomForestClassifier` — the selected model
- `GradientBoostingClassifier` — comparison model
- `SVC` (Support Vector Machine) — comparison model
- `StandardScaler` — normalizes MFCC features before training
- `train_test_split` — splits data into 80% training and 20% testing
- `classification_report`, `confusion_matrix` — evaluation metrics

### SHAP (SHapley Additive exPlanations)
The most important library for the **explainability** component of this project. SHAP calculates how much each feature contributed to a specific prediction — answering not just "is this fake?" but "**why** is this fake?". 

`shap.TreeExplainer` is used specifically because it is optimized for tree-based models (Random Forest, Gradient Boosting) and runs significantly faster than the generic KernelExplainer.

### matplotlib
Used to generate all charts embedded in the PDF report:
- Timeline chart showing which seconds are fake/real
- SHAP feature importance bar chart
- Normal range comparison chart with dots showing actual vs expected values

### ReportLab
A PDF generation library for Python. Used to build the professional PDF report programmatically — including styled text, tables, charts, colored verdict banners, and footer information.

### Streamlit
A Python library that turns Python scripts into web applications with minimal code. Used to build the user-facing web interface where users can upload audio files, see results instantly on screen, and download the PDF report.

### ffmpeg
A command-line audio/video processing tool used to convert MP3 files to WAV format as part of data preparation. Not a Python library — installed separately on the system.

### yt-dlp
A command-line tool used during dataset creation to download audio from YouTube videos. Used to collect real human voice samples from Pakistani talk shows, podcasts, and interviews.

---

## 📊 Section 2 — Dataset

### 2.1 Dataset Composition

| Category | Count | Source |
|----------|-------|--------|
| Real voices | 151 WAV files | YouTube (Pakistani podcasts, interviews, talk shows) |
| Fake voices | 151 WAV files | ElevenLabs, Edge TTS (Microsoft) |
| **Total** | **302 files** | |

After chunking into 3-second segments:

| Category | Chunks |
|----------|--------|
| Real chunks | 536 |
| Fake chunks | 327 |
| **Total** | **863** |

### 2.2 What Makes This Dataset Unique

The dataset specifically contains **code-switched Urdu-English speech** — audio where the speaker mixes both languages naturally within the same sentence. This is the natural speaking style of most urban Pakistanis and is almost completely absent from existing deepfake detection datasets, which are primarily English-only.

Example of code-switched speech in the dataset:
> *"Maine yeh project complete kar liya hai finally, bohot mushkil tha lekin humne deadline se pehle sab kuch implement kar liya"*

### 2.3 Real Voice Collection

Real voices were collected from YouTube using `yt-dlp`. The criteria for selecting videos were:
- Pakistani speaker naturally mixing Urdu and English
- Clear audio with no background music
- Single speaker (no overlapping voices)
- Indoor recording conditions
- Videos from talk shows, podcasts, educational lectures, and interviews

Sources included Geo News, ARY Digital, Pakistani tech podcasts, and university lectures.

### 2.4 Fake Voice Generation

Fake (AI generated) voices were created using various online 
AI text-to-speech and voice generation platforms including 
**ElevenLabs** and several other web-based TTS tools. 
Multiple platforms were used intentionally to ensure the 
dataset contains fake voices with varied synthesis 
characteristics — making the model more robust to different 
types of AI voice generation.

Scripts used for generation covered diverse topics including: computer science discussions, daily life conversations, health and family, education, sports, shopping, and community topics — ensuring the model learns to detect fakes regardless of the topic being discussed.

---

## 🔧 Section 3 — Data Preprocessing

Preprocessing converts raw audio files into a clean, standardized format suitable for machine learning.

### 3.1 MP3 to WAV Conversion

All fake audio files downloaded from ElevenLabs are in MP3 format. The model requires WAV format. Conversion is done using ffmpeg:
```
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav
```
Parameters:
- `-ar 16000` — sets sample rate to 16kHz (standard for speech processing)
- `-ac 1` — converts to mono (removes stereo channel)
- `-sample_fmt s16` — 16-bit audio for high quality

### 3.2 Audio Cleaning

Each audio file goes through a 3-step cleaning process:

**Step 1 — Noise Reduction**
Using `noisereduce`, the first 0.3 seconds of audio are used as a noise profile. The library identifies the frequency pattern of background noise and subtracts it from the entire clip. The `prop_decrease=0.6` parameter means 60% noise reduction — aggressive enough to clean but gentle enough to preserve natural voice characteristics.

**Step 2 — Spike Removal**
Sudden loud sounds (notification pings, mouse clicks, keyboard sounds) appear as energy spikes in the audio. These are detected by calculating RMS energy per frame and identifying frames that exceed `mean + 2 standard deviations`. Detected spikes have their volume reduced to the average level. Gaussian smoothing (`sigma=5`) is applied to the gain mask to prevent audible clicks at transition points.

**Step 3 — Volume Normalization**
All audio clips are normalized to a maximum amplitude of 0.9. This ensures the model is not influenced by loudness differences between samples — a quiet real voice and a loud fake voice should be classified based on voice characteristics, not volume.

### 3.3 Audio Chunking

Each audio file is split into fixed-size 3-second chunks:
- **Why 3 seconds?** Long enough to capture meaningful speech patterns, short enough for efficient processing
- **Why fixed size?** Machine learning models require consistent input dimensions
- **Chunk size:** 48,000 samples (3 seconds × 16,000 Hz)
- Clips shorter than 3 seconds at the end are **discarded** (not padded) to maintain clean data

This chunking increases the dataset from 302 files to 863 chunks — providing more training examples without collecting new data.

### 3.4 Class Imbalance Handling

After chunking, real samples (536) outnumber fake samples (327) due to real voice clips being longer. This imbalance is handled using `class_weight='balanced'` in the Random Forest classifier, which automatically gives more importance to the minority class (fake) during training.

---

## 🤖 Section 4 — Feature Extraction (MFCC)

### 4.1 What are MFCCs?

MFCC stands for **Mel Frequency Cepstral Coefficients**. They are a numerical representation of the short-term power spectrum of audio, designed to mimic how the human auditory system processes sound.

Each 3-second audio chunk is converted into 40 MFCC values — each capturing a different characteristic of the voice:

| MFCC | What it captures |
|------|-----------------|
| MFCC_1 | Overall energy level |
| MFCC_2 | Spectral shape (brightness) |
| MFCC_3 | Voice texture coarseness |
| MFCC_4 | Vocal tract resonance |
| MFCC_5–8 | Mid-level voice characteristics |
| MFCC_9–15 | Fine voice patterns |
| MFCC_16–40 | Ultra-fine spectral details |

### 4.2 Why MFCCs Work for Deepfake Detection

Real human voices have characteristics that AI-generated voices struggle to replicate perfectly:
- Natural micro-variations in pitch and energy
- Subtle breathing artifacts between words
- Glottal pulse patterns from vocal cord vibration
- Natural coarticulation (how sounds blend into each other)

These differences manifest in the MFCC values — especially in higher-order coefficients (MFCC_20 to MFCC_40) that capture fine-grained spectral details.

### 4.3 Feature Extraction Process

```python
audio, sr = librosa.load(file_path, sr=16000, mono=True)
mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
features  = np.mean(mfcc, axis=1)  # → 40 numbers per chunk
```

The `np.mean(mfcc, axis=1)` averages the MFCC values across time, producing one 40-dimensional feature vector per 3-second chunk.

### 4.4 Normalization

Before training, features are normalized using `StandardScaler`:
- Scaler is fitted on training data only (to prevent data leakage)
- Same scaler is applied to test data and any new audio at prediction time
- Normalization ensures all 40 features have equal influence on the model regardless of their original scale

---

## 🧠 Section 5 — Model Training

### 5.1 Train/Test Split

```
Total chunks : 863
Training set : 690 (80%)
Testing set  : 173 (20%)
```

`stratify=y` ensures the real/fake ratio is maintained in both training and testing sets.

### 5.2 Models Compared

Three models were trained and compared on the same dataset:

**Random Forest (Selected)**
An ensemble of 100 decision trees. Each tree votes on the prediction and the majority wins. Ideal for this problem because:
- Works well with the 40-dimensional MFCC feature space
- Naturally handles the class imbalance via `class_weight='balanced'`
- Fully compatible with SHAP TreeExplainer for explainability

**Gradient Boosting**
Builds trees sequentially where each tree corrects the errors of the previous one. Achieved 92.5% accuracy but was slower and incompatible with the SHAP explainability framework used in this project.

**SVM (Support Vector Machine)**
Finds the optimal hyperplane separating real and fake voice patterns in the 40-dimensional feature space. Achieved the highest accuracy (98.8%) but was rejected because SHAP's `TreeExplainer` does not support SVM, and `KernelExplainer` (the SVM alternative) is prohibitively slow for real-time report generation.

### 5.3 Model Selection Decision

```
SVM           → 98.8% accuracy  → rejected (no SHAP support)
Random Forest → 95.4% accuracy  → selected (full SHAP support)
```

This tradeoff — 3.4% accuracy for full explainability — is justified because:
1. Explainability is a core feature of this system (the "why" report)
2. 95.4% is already above industry standard for deepfake detection (85–90%)
3. The system is a research prototype, not a final production system

---

## 📈 Section 6 — Model Evaluation

### 6.1 Classification Report

```
              precision    recall  f1-score   support
        Real       0.95      0.97      0.96       107
        Fake       0.95      0.92      0.94        66
    accuracy                           0.95       173
   macro avg       0.95      0.95      0.95       173
weighted avg       0.95      0.95      0.95       173
```

### 6.2 Confusion Matrix Interpretation

```
Out of 107 real voices tested:
  → 104 correctly identified as REAL ✅
  →   3 incorrectly called FAKE      (false positive)

Out of 66 fake voices tested:
  →  61 correctly identified as FAKE ✅
  →   5 incorrectly called REAL      (false negative)
```

### 6.3 Metric Explanations

- **Precision (95%)** — Of all voices the model labeled as fake, 95% were actually fake
- **Recall (92%)** — Of all actual fake voices, the model correctly caught 92%
- **F1 Score (0.94)** — Balanced measure combining precision and recall
- **Accuracy (95.4%)** — Overall percentage of correct predictions

---

## 📋 Section 7 — Explainability (SHAP)

### 7.1 Why Explainability Matters

A system that only says "this is fake" is not useful in practice. Decision makers (security teams, journalists, courts) need to know **why** a voice is considered fake. SHAP provides this explanation.

### 7.2 How SHAP Works

SHAP (SHapley Additive exPlanations) is based on game theory. It assigns each feature a value indicating how much it contributed to the prediction for a specific audio sample:

- **Positive SHAP value** → Feature pushed the model toward FAKE
- **Negative SHAP value** → Feature pushed the model toward REAL
- **Larger absolute value** → Stronger influence on the decision

### 7.3 Reading the SHAP Chart

The SHAP chart in the PDF report shows the top 12 most influential features:

```
Red bars  → features pushing toward FAKE
Green bars → features pushing toward REAL
Bar length → strength of influence
```

Even in a correctly detected fake audio, some features may show green bars — meaning those specific features appear normal. The model considers all 40 features together, and the overall balance determines the final verdict.

### 7.4 Normal Range Comparison

Alongside the SHAP chart, a normal range chart shows where each feature falls relative to real human voices:

- **Blue bar** — the normal range (mean ± 1 standard deviation) for that feature across all real voices in training data
- **Colored dot** — where this specific audio sample falls
- **Dot inside blue** → Within normal human range
- **Dot outside blue** → Abnormal — suspicious feature

### 7.5 Why Features Can Be Within Range But Still Fake

A common question: if most features are within normal range, why is the audio predicted as fake?

The answer is **combination patterns**. The model does not evaluate each feature in isolation — it evaluates all 40 features together. An AI voice may have each individual feature within normal range, but the *specific combination* of values across all 40 features creates a pattern that real human voices never produce naturally.

Think of it like a fingerprint — each ridge may look normal individually, but the overall pattern is unique and identifiable.

---

## 🖥️ Section 8 — Web Application (Streamlit)

### 8.1 What the App Does

The Streamlit web app provides a user-friendly interface for non-technical users to analyze audio files without running any code.

**User flow:**
1. Open the web app in any browser
2. Upload a WAV or MP3 audio file
3. Wait a few seconds for analysis
4. See the verdict (REAL/FAKE) with confidence score
5. View the timeline chart showing suspicious segments
6. View the SHAP feature analysis chart
7. See segment-by-segment breakdown
8. Download the full PDF report

### 8.2 Running the App Locally

```bash
pip install streamlit
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`

### 8.3 Deploying to Streamlit Cloud (Free)

1. Push the project to GitHub (without the dataset folder)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and `app.py` as the main file
5. Click Deploy — get a public URL instantly

---

## 🔄 Section 9 — Pipeline Overview

The complete end-to-end pipeline:

```
Raw Audio File
      ↓
1. LOAD (librosa — 16kHz mono)
      ↓
2. CLEAN
   ├── Noise reduction (noisereduce)
   ├── Spike removal (RMS energy threshold)
   └── Volume normalization
      ↓
3. CHUNK (split into 3-second segments)
      ↓
4. EXTRACT FEATURES (40 MFCC values per chunk)
      ↓
5. NORMALIZE (StandardScaler)
      ↓
6. PREDICT (Random Forest — 95.4% accuracy)
      ↓
7. MAJORITY VOTE (across all chunks)
      ↓
8. EXPLAIN (SHAP values for top 12 features)
      ↓
9. GENERATE REPORT (PDF with verdict + charts)
```

---

## 📁 Section 10 — Notebooks Guide

### 01_data_preparation.ipynb
Run this first — only needs to be run once when setting up the project or adding new data.
- Converts any MP3 files to WAV format
- Updates metadata.csv with all file labels
- Splits all audio into 3-second chunks
- Saves chunks to `dataset/real_chunks/` and `dataset/fake_chunks/`

### 02_model_training.ipynb
Run this second — trains the model and saves it.
- Loads chunk files from dataset folders
- Extracts MFCC features from all 863 chunks
- Splits into 80/20 train/test sets
- Trains and compares Random Forest, Gradient Boosting, and SVM
- Evaluates and shows confusion matrix
- Saves `model.pkl`, `scaler.pkl`, `real_means.npy`, `real_stds.npy`

### 03_report_generation.ipynb
Run this to analyze any audio file.
- Loads the saved model and scaler
- Change `TEST_FILE` to any WAV file path
- Cleans, chunks, and predicts the audio
- Generates timeline and SHAP charts
- Builds and saves PDF report to `audio_reports/`

### 04_live_detection.ipynb
Run this for real-time microphone detection.
- Loads the saved model and scaler
- Records 3-second audio clips from the microphone continuously
- Predicts each clip in real time
- Prints REAL/FAKE result with confidence after each clip
- Press Ctrl+C to stop

---

## ⚙️ Section 11 — Installation and Setup

### Prerequisites
- Python 3.10 or higher
- ffmpeg installed and added to system PATH
- Microphone (for live detection only)

### Installation

```bash
# Clone the repository
git clone https://github.com/UeMHRB/Pakistani-Deepfake-Voice-Detection
cd deepfake-voice-detection

# Install required libraries
pip install -r requirements.txt
```

### requirements.txt

```
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
shap>=0.42.0
matplotlib>=3.7.0
noisereduce>=3.0.0
scipy>=1.11.0
reportlab>=4.0.0
streamlit>=1.28.0
sounddevice>=0.4.6
```

### Dataset Setup

The dataset is not included in this repository due to size constraints. To set up the dataset:

1. Create the folder structure:
```
dataset/
    real/     ← place real WAV files here
    fake/     ← place fake WAV files here
```

2. Run `01_data_preparation.ipynb` to process the data

3. Run `02_model_training.ipynb` to train the model

Alternatively, pre-trained model files (`model.pkl`, `scaler.pkl`, `real_means.npy`, `real_stds.npy`) can be downloaded from the project's Google Drive link (see Releases section).

---

## 🔬 Section 12 — Limitations and Future Work

### Current Limitations

**1. Distribution Shift**
The model performs best on audio similar to the training data (Pakistani YouTube videos, ElevenLabs/Edge TTS generated audio). Audio from very different recording environments or with unusual acoustic characteristics may produce uncertain predictions (confidence near 50–65%).

**2. Dataset Size**
With 863 training chunks, the model is well-trained for a research prototype but would benefit from a larger and more diverse dataset for production deployment.

**3. Language Coverage**
While the system handles Urdu-English code-switching well, it has not been tested on other Pakistani languages such as Punjabi, Sindhi, or Pashto.

**4. Fake Voice Generator Coverage**
The fake samples were generated using ElevenLabs and several other web-based TTS platforms Additional TTS tools not represented in the training data may produce different 
synthesis artifacts that the model has not yet learned to detect.

### Future Work

- Expand dataset to include more diverse speakers, recording conditions, and languages
- Add support for detecting deepfakes from additional TTS tools
- Implement a transformer-based feature extractor (wav2vec 2.0) for better generalization
- Add real-time call monitoring integration
- Build a mobile application
- Train separate models for Urdu-only and English-only speech for comparison

---

## 👥 Team

| Name | Role |
|------|------|
| Muhammad Hamza | Data collection, model training, report generation, , web app |
| Shafia Tooba | Dataset preparation, fake voice generation,model training, report generation |

**Institution:** NETSOL Institute of Artificial Intelligence (NIAI)
**Program:** NAVTTC AI/ML Deep Learning Course
**Duration:** 3 months

---

## 📚 References

- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing.*
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems.*
- Breiman, L. (2001). Random forests. *Machine Learning.*
- ElevenLabs. (2023). Voice AI platform. https://elevenlabs.io
- McFee, B., et al. (2015). librosa: Audio and music signal analysis in Python. *Proceedings of the 14th Python in Science Conference.*

---

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 🙏 Acknowledgements

- **NAVTTC** (National Vocational and Technical Training Commission) for providing the training opportunity
- **NETSOL Technologies** for hosting the NIAI program
- The open source community behind librosa, scikit-learn, SHAP, and Streamlit

---

*Built with ❤️ in Lahore, Pakistan*
