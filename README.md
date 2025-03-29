# Speech-Understanding-Programming-Assignment---2

Speaker Separation, Identification, and Enhancement Pipeline

Speaker Separation, Identification, and Enhancement Pipeline

Overview
This project implements a novel pipeline, SepID-Enhance, combining speaker separation (using SepFormer), speaker identification (using WavLM Base Plus), and speech enhancement. Additionally, it includes a language classification module using MFCC features extracted from the "Audio Dataset with 10 Indian Languages." The pipeline is fine-tuned on a custom multi-speaker dataset derived from VoxCeleb2 and evaluated on various metrics.


Features:

Speaker Separation & Enhancement:
Uses SepFormer (speechbrain/sepformer-wsj02mix) for separating mixed audio into individual speaker streams.
Enhances speech quality through joint training with identification feedback.

Speaker Identification:
Employs pre-trained and fine-tuned WavLM Base Plus (microsoft/wavlm-base-plus) with LoRA and ArcFace loss.
Identifies speakers in separated audio streams.

Language Classification:
Extracts MFCC features from the "Audio Dataset with 10 Indian Languages" (Kaggle).
Trains a Random Forest Classifier to predict the language of audio samples.

Prerequisites:

Python: 3.8+
Dependencies: Install via pip:

torch torchaudio transformers speechbrain pesq numpy tqdm peft librosa soundfile scikit-learn matplotlib

Datasets:

VoxCeleb2: Download from VoxCeleb and place at /path/to/voxceleb2/vox2.
Indian Languages Audio Dataset: Download from Kaggle using:

