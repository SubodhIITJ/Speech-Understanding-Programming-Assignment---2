# -*- coding: utf-8 -*-
"""M23CSA531_PA2
"""

from google.colab import drive
drive.mount('/content/drive')

"""## **Question 1: Speech Enhancement**

# Q. II
"""

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from torch import nn
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""**pre-trained Model**"""

# Load pre-trained WavLM Base Plus model and feature extractor
model_name = "microsoft/wavlm-base-plus"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMModel.from_pretrained(model_name).to(device)
model.eval()

# Function to load and preprocess audio
def load_audio(file_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
    return waveform.squeeze(0)  # Remove channel dimension if mono

# Function to extract embeddings
def extract_embedding(audio_path):
    waveform = load_audio(audio_path)
    # Process audio with feature extractor
    inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)

    with torch.no_grad():
        outputs = model(input_values)
        # Use the mean of the last hidden state as the embedding
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Cosine similarity function
cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

# Paths
voxceleb_root = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/wav"
trial_file = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/VoxCeleb1-cleaned.txt"

# Load trial pairs
trials = []
with open(trial_file, "r") as f:
    for line in f:
        label, file1, file2 = line.strip().split()
        trials.append((int(label), file1, file2))

# Dictionary to cache embeddings
embedding_cache = {}

# Compute similarity scores
scores = []
labels = []
for label, file1, file2 in tqdm(trials[:1000]):  
    file1_path = os.path.join(voxceleb_root, file1)
    file2_path = os.path.join(voxceleb_root, file2)

    # Verify file existence
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print(f"Skipping missing file: {file1_path} or {file2_path}")
        continue

    # Get embeddings (cache to avoid recomputation)
    if file1_path not in embedding_cache:
        embedding_cache[file1_path] = extract_embedding(file1_path)
    if file2_path not in embedding_cache:
        embedding_cache[file2_path] = extract_embedding(file2_path)

    emb1 = torch.from_numpy(embedding_cache[file1_path]).to(device)
    emb2 = torch.from_numpy(embedding_cache[file2_path]).to(device)

    score = cosine_similarity(emb1, emb2).item()
    scores.append(score)
    labels.append(label)

# Metric 1: EER (in %)
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer = interp1d(fpr, fnr)(eer_threshold)
    return eer * 100

eer = compute_eer(labels, scores)
print(f"Equal Error Rate (EER): {eer:.2f}%")

# Metric 2: TAR@1%FAR
def compute_tar_at_far(labels, scores, target_far=0.01):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    tar_at_far = interp1d(fpr, tpr)(target_far)
    return tar_at_far * 100

tar_at_1far = compute_tar_at_far(labels, scores, target_far=0.01)
print(f"TAR@1%FAR: {tar_at_1far:.2f}%")

# Metric 3: Speaker Identification Accuracy
def compute_identification_accuracy(labels, scores, threshold=0.5):
    predictions = [1 if score >= threshold else 0 for score in scores]
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    accuracy = correct / len(labels) * 100  # Convert to percentage
    return accuracy

# Use EER threshold for identification (optional: tune this)
fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
id_accuracy = compute_identification_accuracy(labels, scores, threshold=eer_threshold)
print(f"Speaker Identification Accuracy: {id_accuracy:.2f}%")

# Save scores and labels for further analysis
np.save("/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/scores.npy", np.array(scores))
np.save("/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/labels.npy", np.array(labels))

"""**fine-tune Model**

Lets fine-tune the microsoft/wavlm-base-plus model for speaker verification using LoRA (Low-Rank Adaptation) and ArcFace loss on the VoxCeleb2 dataset.
"""

from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ArcFace Loss Implementation
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * (theta + self.m) + (1.0 - one_hot) * theta).cos() * self.s
        return F.cross_entropy(output, labels)

# Custom Dataset with padding/truncation
class VoxCeleb2Dataset(Dataset):
    def __init__(self, files, max_length=48000):  # 3 seconds at 16kHz
        self.files = files
        self.max_length = max_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, speaker_id = self.files[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        waveform = waveform.squeeze(0)


        if waveform.size(0) > self.max_length:
            waveform = waveform[:self.max_length]
        elif waveform.size(0) < self.max_length:
            padding = torch.zeros(self.max_length - waveform.size(0))
            waveform = torch.cat([waveform, padding])

        return waveform, speaker_id

# Load pre-trained model and feature extractor
model_name = "microsoft/wavlm-base-plus"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
model = WavLMModel.from_pretrained(model_name).to(device)

# Apply LoRA
lora_config = LoraConfig(
    r=32,  # Increased rank
    lora_alpha=32,
    target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"],
    lora_dropout=0.1
)
model = get_peft_model(model, lora_config)
model.train()

# Paths
voxceleb2_root = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/vox2/aac"
voxceleb1_trial_file = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/VoxCeleb1-cleaned.txt"
voxceleb1_root = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/wav"

# Load VoxCeleb2 identities
all_ids = sorted([d for d in os.listdir(voxceleb2_root) if os.path.isdir(os.path.join(voxceleb2_root, d))])[:118]
train_ids = all_ids[:100]
test_ids = all_ids[100:]

# Prepare training data
train_files = []
for speaker_id in train_ids:
    speaker_path = os.path.join(voxceleb2_root, speaker_id)
    for session in os.listdir(speaker_path):
        session_path = os.path.join(speaker_path, session)
        files = [f for f in os.listdir(session_path) if f.endswith((".wav", ".m4a"))]
        for file in files:
            train_files.append((os.path.join(session_path, file), speaker_id))

print(f"Collected {len(train_files)} training files from {len(train_ids)} speakers.")

# Fine-tuning setup
train_dataset = VoxCeleb2Dataset(train_files[:5000])  # Larger subset
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
arcface_loss = ArcFaceLoss(in_features=768, out_features=len(train_ids)).to(device)
id_to_idx = {id: idx for idx, id in enumerate(train_ids)}

# Training loop
for epoch in range(5):
    total_loss = 0
    for waveforms, speaker_ids in tqdm(train_loader):

        inputs = feature_extractor(waveforms.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device)

        optimizer.zero_grad()
        outputs = model(input_values).last_hidden_state.mean(dim=1)
        labels = torch.tensor([id_to_idx[sid] for sid in speaker_ids], dtype=torch.long).to(device)
        loss = arcface_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

model.eval()

# Evaluation function
def extract_embedding(audio_path, model):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    waveform = waveform.squeeze(0)
    inputs = feature_extractor(waveform.tolist(), sampling_rate=16e3, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        outputs = model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

# Load VoxCeleb1 trial pairs
trials = []
with open(voxceleb1_trial_file, "r") as f:
    for line in f:
        label, file1, file2 = line.strip().split()
        trials.append((int(label), file1, file2))

# Evaluate pre-trained and fine-tuned models
def evaluate_model(model, name):
    embedding_cache = {}
    scores = []
    labels = []
    trial_subset = trials[:1000]
    for label, file1, file2 in tqdm(trial_subset):
        file1_path = os.path.join(voxceleb1_root, file1)
        file2_path = os.path.join(voxceleb1_root, file2)

        if not os.path.exists(file1_path) or not os.path.exists(file2_path):
            print(f"Skipping missing file: {file1_path} or {file2_path}")
            continue

        if file1_path not in embedding_cache:
            embedding_cache[file1_path] = extract_embedding(file1_path, model)
        if file2_path not in embedding_cache:
            embedding_cache[file2_path] = extract_embedding(file1_path, model)

        emb1 = torch.from_numpy(embedding_cache[file1_path]).to(device)
        emb2 = torch.from_numpy(embedding_cache[file2_path]).to(device)
        score = cosine_similarity(emb1, emb2).item()
        scores.append(score)
        labels.append(label)

    if not labels:
        print(f"No valid trial pairs processed for {name}. Check voxceleb1_root and trial file paths.")
        return None, None, None

    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100
    eer_threshold = thresholds[np.argmin(np.abs(fpr - (1 - tpr)))]
    tar_at_1far = interp1d(fpr, tpr)(0.01) * 100
    predictions = [1 if score >= eer_threshold else 0 for score in scores]
    id_accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels) * 100

    print(f"{name} - EER: {eer:.2f}%, TAR@1%FAR: {tar_at_1far:.2f}%, Speaker ID Accuracy: {id_accuracy:.2f}%")
    return eer, tar_at_1far, id_accuracy

# Load pre-trained model for comparison
pretrained_model = WavLMModel.from_pretrained(model_name).to(device)
pretrained_model.eval()

# Evaluate both models
pretrained_metrics = evaluate_model(pretrained_model, "Pre-trained")
finetuned_metrics = evaluate_model(model, "Fine-tuned")

"""# Q. III A , Step 1: Create the Multi-Speaker Dataset"""

import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
import random

# Paths
voxceleb2_root = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/vox2/aac"
output_train_dir = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/output/train_mixtures"
output_test_dir = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/output/test_mixtures"
os.makedirs(output_train_dir, exist_ok=True)
os.makedirs(output_test_dir, exist_ok=True)

# Load VoxCeleb2 identities (sorted ascending)
all_ids = sorted([d for d in os.listdir(voxceleb2_root) if os.path.isdir(os.path.join(voxceleb2_root, d))])
train_ids = all_ids[:50]  # First 50 for training
test_ids = all_ids[50:100]  # Next 50 for testing

# Function to load and resample audio
def load_audio(file_path, target_sr=16000):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sr)(waveform)
    return waveform.squeeze(0)
# Function to mix two utterances
def mix_utterances(file1, file2, max_length=48000):  # 3 seconds
    wav1 = load_audio(file1)
    wav2 = load_audio(file2)

    # Truncate or pad to max_length
    if wav1.size(0) > max_length:
        wav1 = wav1[:max_length]
    elif wav1.size(0) < max_length:
        wav1 = torch.cat([wav1, torch.zeros(max_length - wav1.size(0))])

    if wav2.size(0) > max_length:
        wav2 = wav2[:max_length]
    elif wav2.size(0) < max_length:
        wav2 = torch.cat([wav2, torch.zeros(max_length - wav2.size(0))])

    # Mix with random gain between 0.5 and 1.0
    gain1, gain2 = random.uniform(0.5, 1.0), random.uniform(0.5, 1.0)
    mixture = gain1 * wav1 + gain2 * wav2
    mixture = mixture / torch.max(torch.abs(mixture))

    return mixture, wav1, wav2

# Collect files for each identity
def collect_files(ids, root_dir):
    files_dict = {}
    for speaker_id in ids:
        speaker_path = os.path.join(root_dir, speaker_id)
        files = []
        for session in os.listdir(speaker_path):
            session_path = os.path.join(speaker_path, session)
            files.extend([os.path.join(session_path, f) for f in os.listdir(session_path) if f.endswith(".m4a")])
        files_dict[speaker_id] = files
    return files_dict

# Create mixtures
def create_mixtures(ids, files_dict, output_dir, num_mixtures=100):
    for i in tqdm(range(num_mixtures)):
        # Randomly select two different speakers
        spk1, spk2 = random.sample(ids, 2)
        file1 = random.choice(files_dict[spk1])
        file2 = random.choice(files_dict[spk2])

        mixture, wav1, wav2 = mix_utterances(file1, file2)

        # Save mixture and original sources
        torchaudio.save(os.path.join(output_dir, f"mix_{i}.wav"), mixture.unsqueeze(0), 16000)
        torchaudio.save(os.path.join(output_dir, f"src1_{i}.wav"), wav1.unsqueeze(0), 16000)
        torchaudio.save(os.path.join(output_dir, f"src2_{i}.wav"), wav2.unsqueeze(0), 16000)

# Generate datasets
train_files = collect_files(train_ids, voxceleb2_root)
test_files = collect_files(test_ids, voxceleb2_root)
create_mixtures(train_ids, train_files, output_train_dir, num_mixtures=100)  # 100 training mixtures
create_mixtures(test_ids, test_files, output_test_dir, num_mixtures=50)    # 50 testing mixtures

"""# Q. III A , Step 2: Speaker Separation with SepFormer"""

import torch
import torchaudio
from speechbrain.pretrained import SepformerSeparation
from pesq import pesq
from pystoi import stoi
import numpy as np
from tqdm import tqdm
import os

# Load pre-trained SepFormer model
model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

# Evaluation metrics functions
def compute_sdr(ref, est):
    """Simplified SDR calculation"""
    s_target = ref
    e_noise = est - ref
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_noise**2) + 1e-8))

def compute_sir(ref, est, interferer):
    """Simplified SIR calculation"""
    s_target = ref
    e_interf = interferer
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_interf**2) + 1e-8))

def compute_sar(ref, est):
    """Simplified SAR calculation"""
    s_target = ref
    e_artifacts = est - ref
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_artifacts**2) + 1e-8))

# Paths
test_dir = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/output/test_mixtures"

# Evaluate on test set
results = {"SIR": [], "SAR": [], "SDR": [], "PESQ": []}
for i in tqdm(range(50)):  # 50 test mixtures
    mix_path = os.path.join(test_dir, f"mix_{i}.wav")
    src1_path = os.path.join(test_dir, f"src1_{i}.wav")
    src2_path = os.path.join(test_dir, f"src2_{i}.wav")

    # Load mixture and references
    mixture, sr = torchaudio.load(mix_path)
    ref1, _ = torchaudio.load(src1_path)
    ref2, _ = torchaudio.load(src2_path)
    mixture = mixture.squeeze(0).numpy()
    ref1 = ref1.squeeze(0).numpy()
    ref2 = ref2.squeeze(0).numpy()

    print(f"Mixture length: {mixture.shape[0]} samples ({mixture.shape[0]/16000:.2f}s)")

    # Perform separation
    est_sources = model.separate_file(mix_path)
    est_sources = est_sources.squeeze(0).detach().cpu().numpy()
    print(f"Est sources shape: {est_sources.shape}")

    # Validate shape
    if len(est_sources.shape) != 2 or est_sources.shape[1] != 2:
        raise ValueError(f"Expected [samples, 2], got {est_sources.shape}")

    est1, est2 = est_sources[:, 0], est_sources[:, 1]
    print(f"Est1 shape: {est1.shape}, Est2 shape: {est2.shape}")

    # Adjust lengths to match estimated sources
    min_len = min(est1.shape[0], ref1.shape[0])
    est1, est2 = est1[:min_len], est2[:min_len]
    ref1, ref2 = ref1[:min_len], ref2[:min_len]
    print(f"Adjusted lengths to {min_len} samples ({min_len/16000:.2f}s)")

    # Compute metrics
    sir1 = compute_sir(ref1, est1, ref2)
    sir2 = compute_sir(ref2, est2, ref1)
    sar1 = compute_sar(ref1, est1)
    sar2 = compute_sar(ref2, est2)
    sdr1 = compute_sdr(ref1, est1)
    sdr2 = compute_sdr(ref2, est2)
    pesq1 = pesq(16000, ref1, est1, "wb")
    pesq2 = pesq(16000, ref2, est2, "wb")

    # Store results
    results["SIR"].extend([sir1, sir2])
    results["SAR"].extend([sar1, sar2])
    results["SDR"].extend([sdr1, sdr2])
    results["PESQ"].extend([pesq1, pesq2])

# Compute averages
for metric in results:
    avg = np.mean(results[metric])
    print(f"Average {metric}: {avg:.2f}")

"""# Q. III B"""

pip install speechbrain

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from speechbrain.inference import SepformerSeparation
import numpy as np
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained WavLM and feature extractor
model_name = "microsoft/wavlm-base-plus"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
pretrained_model = WavLMModel.from_pretrained(model_name).to(device)
pretrained_model.eval()

# Load fine-tuned WavLM (assuming saved from first task)
finetuned_model = WavLMModel.from_pretrained(model_name).to(device)
lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"],
    lora_dropout=0.1
)

finetuned_model = get_peft_model(finetuned_model, lora_config)
# Load fine-tuned weights (update path to your saved model)
finetuned_model.load_state_dict(torch.load("/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/finetuned_model.pth"))
finetuned_model.eval()

# Load SepFormer model
sep_model = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-wsj02mix",
    savedir="pretrained_models/sepformer-wsj02mix"
)

# Paths
test_dir = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/output/test_mixtures"
voxceleb2_root = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/vox2/aac"

# Test identities (50-99)
all_ids = sorted([d for d in os.listdir(voxceleb2_root) if os.path.isdir(os.path.join(voxceleb2_root, d))])
test_ids = all_ids[50:100]
id_to_idx = {id: idx for idx, id in enumerate(test_ids)}

# Function to extract embedding
def extract_embedding(waveform, model):
    inputs = feature_extractor(waveform.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        outputs = model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Cosine similarity
cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

# Collect reference embeddings for test identities
ref_embeddings_pretrained = {}
ref_embeddings_finetuned = {}
for speaker_id in test_ids:
    speaker_path = os.path.join(voxceleb2_root, speaker_id, os.listdir(os.path.join(voxceleb2_root, speaker_id))[0])
    file = os.path.join(speaker_path, os.listdir(speaker_path)[0])
    waveform, sr = torchaudio.load(file)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.squeeze(0).numpy()

    ref_embeddings_pretrained[speaker_id] = extract_embedding(waveform, pretrained_model)
    ref_embeddings_finetuned[speaker_id] = extract_embedding(waveform, finetuned_model)

# Evaluate on separated test set
correct_pretrained = 0
correct_finetuned = 0
total = 0

for i in tqdm(range(50)):  # 50 test mixtures
    mix_path = os.path.join(test_dir, f"mix_{i}.wav")
    src1_path = os.path.join(test_dir, f"src1_{i}.wav")
    src2_path = os.path.join(test_dir, f"src2_{i}.wav")

    # Ground truth speaker IDs
    src1_file = os.path.basename(src1_path).split("_")[1]
    src2_file = os.path.basename(src2_path).split("_")[1]
    true_id1 = os.path.basename(os.path.dirname(os.path.dirname(src1_path)))
    true_id2 = os.path.basename(os.path.dirname(os.path.dirname(src2_path)))

    # Separation
    est_sources = sep_model.separate_file(mix_path).squeeze(0).detach().cpu().numpy()
    est1, est2 = est_sources[:, 0], est_sources[:, 1]

    # Extract embeddings from separated sources
    emb1_pretrained = extract_embedding(est1, pretrained_model)
    emb2_pretrained = extract_embedding(est2, pretrained_model)
    emb1_finetuned = extract_embedding(est1, finetuned_model)
    emb2_finetuned = extract_embedding(est2, finetuned_model)

    # Compute similarities and predict speakers
    pretrained_scores = {}
    finetuned_scores = {}
    for speaker_id in test_ids:
        ref_pre = torch.from_numpy(ref_embeddings_pretrained[speaker_id]).to(device)
        ref_fin = torch.from_numpy(ref_embeddings_finetuned[speaker_id]).to(device)
        pretrained_scores[speaker_id] = [
            cosine_similarity(torch.from_numpy(emb1_pretrained).to(device), ref_pre).item(),
            cosine_similarity(torch.from_numpy(emb2_pretrained).to(device), ref_pre).item()
        ]
        finetuned_scores[speaker_id] = [
            cosine_similarity(torch.from_numpy(emb1_finetuned).to(device), ref_fin).item(),
            cosine_similarity(torch.from_numpy(emb2_finetuned).to(device), ref_fin).item()
        ]

    # Rank-1 prediction
    pred_id1_pre = max(pretrained_scores, key=lambda k: pretrained_scores[k][0])
    pred_id2_pre = max(pretrained_scores, key=lambda k: pretrained_scores[k][1])
    pred_id1_fin = max(finetuned_scores, key=lambda k: finetuned_scores[k][0])
    pred_id2_fin = max(finetuned_scores, key=lambda k: finetuned_scores[k][1])

    # Check correctness (permutation invariant)
    pre_correct = (pred_id1_pre == true_id1 and pred_id2_pre == true_id2) or \
                  (pred_id1_pre == true_id2 and pred_id2_pre == true_id1)
    fin_correct = (pred_id1_fin == true_id1 and pred_id2_fin == true_id2) or \
                  (pred_id1_fin == true_id2 and pred_id2_fin == true_id1)

    correct_pretrained += pre_correct
    correct_finetuned += fin_correct
    total += 1

# Compute Rank-1 accuracy
rank1_acc_pretrained = correct_pretrained / total * 100
rank1_acc_finetuned = correct_finetuned / total * 100

print(f"Pre-trained WavLM Rank-1 Accuracy: {rank1_acc_pretrained:.2f}%")
print(f"Fine-tuned WavLM Rank-1 Accuracy: {rank1_acc_finetuned:.2f}%")

"""# Q. IV A,B"""

pip install pesq

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from speechbrain.pretrained import SepformerSeparation
import numpy as np
from tqdm import tqdm
import os
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from pesq import pesq

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
voxceleb2_root = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/vox2/aac"
train_dir = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/output/train_mixtures"
test_dir = "/content/drive/MyDrive/Colab Notebooks/SEM03-Assignments/Speech Understanding/Assignment2/output/test_mixtures"


# Load models
model_name = "microsoft/wavlm-base-plus"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
pretrained_wavlm = WavLMModel.from_pretrained(model_name).to(device)
pretrained_wavlm.eval()

# Fine-tuned WavLM with LoRA
finetuned_wavlm = WavLMModel.from_pretrained(model_name).to(device)
lora_config = LoraConfig(r=32, lora_alpha=32, target_modules=["attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"], lora_dropout=0.1)
finetuned_wavlm = get_peft_model(finetuned_wavlm, lora_config)

# SepFormer
sepformer = SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="pretrained_models/sepformer-wsj02mix").to(device)

# Dataset
class MultiSpeakerDataset(Dataset):
    def __init__(self, data_dir, max_length=48000):
        self.data_dir = data_dir
        self.max_length = max_length
        self.files = [f for f in os.listdir(data_dir) if f.startswith("mix_") and f.endswith(".wav")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mix_path = os.path.join(self.data_dir, self.files[idx])
        src1_path = os.path.join(self.data_dir, f"src1_{idx}.wav")
        src2_path = os.path.join(self.data_dir, f"src2_{idx}.wav")

        mix, sr = torchaudio.load(mix_path)
        src1, _ = torchaudio.load(src1_path)
        src2, _ = torchaudio.load(src2_path)

        if sr != 16000:
            mix = torchaudio.transforms.Resample(sr, 16000)(mix)
            src1 = torchaudio.transforms.Resample(sr, 16000)(src1)
            src2 = torchaudio.transforms.Resample(sr, 16000)(src2)

        mix, src1, src2 = mix.squeeze(0), src1.squeeze(0), src2.squeeze(0)
        if mix.size(0) > self.max_length:
            mix, src1, src2 = mix[:self.max_length], src1[:self.max_length], src2[:self.max_length]
        elif mix.size(0) < self.max_length:
            padding = torch.zeros(self.max_length - mix.size(0))
            mix = torch.cat([mix, padding])
            src1 = torch.cat([src1, padding])
            src2 = torch.cat([src2, padding])

        # Extract IDs from filenames (assuming format src1_idXXXXX_idx.wav)
        id1 = src1_path.split("src1_")[1].split("_")[0] if "src1_" in src1_path else os.path.basename(os.path.dirname(os.path.dirname(src1_path)))
        id2 = src2_path.split("src2_")[1].split("_")[0] if "src2_" in src2_path else os.path.basename(os.path.dirname(os.path.dirname(src2_path)))
        return mix, src1, src2, id1, id2

# Load datasets
train_dataset = MultiSpeakerDataset(train_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataset = MultiSpeakerDataset(test_dir)

# Identification loss
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(cosine).scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * (theta + self.m) + (1.0 - one_hot) * theta).cos() * self.s
        return F.cross_entropy(output, labels)

# Training setup
train_ids = sorted([d for d in os.listdir(voxceleb2_root) if os.path.isdir(os.path.join(voxceleb2_root, d))])[:50]
test_ids = sorted([d for d in os.listdir(voxceleb2_root) if os.path.isdir(os.path.join(voxceleb2_root, d))])[50:100]
id_to_idx = {id: idx for idx, id in enumerate(train_ids)}
optimizer = torch.optim.Adam(list(sepformer.parameters()) + list(finetuned_wavlm.parameters()), lr=1e-4)
arcface_loss = ArcFaceLoss(in_features=768, out_features=len(train_ids)).to(device)
cosine_similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

# Fine-tuning loop
def train_pipeline():
    sepformer.train()
    finetuned_wavlm.train()
    for epoch in range(5):
        total_loss = 0
        for mix, src1, src2, id1, id2 in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            mix, src1, src2 = mix.to(device), src1.to(device), src2.to(device)
            print(f"ID1: {id1}, ID2: {id2}")  # Debug
            labels = torch.tensor([id_to_idx[i] for i in id1] + [id_to_idx[i] for i in id2], dtype=torch.long).to(device)

            optimizer.zero_grad()
            est_sources = sepformer(mix.unsqueeze(1))  # [batch, samples, 2]
            print(f"SepFormer output shape: {est_sources.shape}")
            est1, est2 = est_sources[..., 0], est_sources[..., 1]

            inputs1 = feature_extractor(est1.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
            inputs2 = feature_extractor(est2.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
            emb1 = finetuned_wavlm(inputs1["input_values"].to(device)).last_hidden_state.mean(dim=1)
            emb2 = finetuned_wavlm(inputs2["input_values"].to(device)).last_hidden_state.mean(dim=1)
            embeddings = torch.cat([emb1, emb2], dim=0)

            sep_loss = -torch.mean(torch.tensor([compute_sdr(src1[i].cpu().numpy(), est1[i].cpu().numpy()) +
                                                compute_sdr(src2[i].cpu().numpy(), est2[i].cpu().numpy())
                                                for i in range(mix.size(0))], requires_grad=True).to(device))
            id_loss = arcface_loss(embeddings, labels)
            loss = sep_loss + 0.1 * id_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(train_loader):.4f}")

# Metric functions
def compute_sdr(ref, est):
    s_target = ref
    e_noise = est - ref
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_noise**2) + 1e-8))

def compute_sir(ref, est, interferer):
    s_target = ref
    e_interf = interferer
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_interf**2) + 1e-8))

def compute_sar(ref, est):
    s_target = ref
    e_artifacts = est - ref
    return 10 * np.log10(np.mean(s_target**2) / (np.mean(e_artifacts**2) + 1e-8))

# Evaluation
def evaluate_pipeline():
    sepformer.eval()
    pretrained_wavlm.eval()
    finetuned_wavlm.eval()
    results = {"SIR": [], "SAR": [], "SDR": [], "PESQ": []}
    correct_pre, correct_fin, total = 0, 0, 0

    ref_emb_pre, ref_emb_fin = {}, {}
    for speaker_id in test_ids:
        speaker_path = os.path.join(voxceleb2_root, speaker_id, os.listdir(os.path.join(voxceleb2_root, speaker_id))[0])
        file = os.path.join(speaker_path, os.listdir(speaker_path)[0])
        waveform, sr = torchaudio.load(file)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = waveform.squeeze(0).numpy()
        ref_emb_pre[speaker_id] = extract_embedding(waveform, pretrained_wavlm)
        ref_emb_fin[speaker_id] = extract_embedding(waveform, finetuned_wavlm)

    with torch.no_grad():
        for i in tqdm(range(len(test_dataset)), desc="Evaluating"):
            mix, src1, src2, id1, id2 = test_dataset[i]
            mix = mix.unsqueeze(0).to(device)
            src1, src2 = src1.numpy(), src2.numpy()

            est_sources = sepformer(mix.unsqueeze(1)).squeeze(0).cpu().numpy()
            est1, est2 = est_sources[:, 0], est_sources[:, 1]

            min_len = min(est1.shape[0], src1.shape[0])
            est1, est2 = est1[:min_len], est2[:min_len]
            src1, src2 = src1[:min_len], src2[:min_len]

            results["SIR"].extend([compute_sir(src1, est1, src2), compute_sir(src2, est2, src1)])
            results["SAR"].extend([compute_sar(src1, est1), compute_sar(src2, est2)])
            results["SDR"].extend([compute_sdr(src1, est1), compute_sdr(src2, est2)])
            results["PESQ"].extend([pesq(16000, src1, est1, "wb"), pesq(16000, src2, est2, "wb")])

            emb1_pre = extract_embedding(est1, pretrained_wavlm)
            emb2_pre = extract_embedding(est2, pretrained_wavlm)
            emb1_fin = extract_embedding(est1, finetuned_wavlm)
            emb2_fin = extract_embedding(est2, finetuned_wavlm)

            pre_scores, fin_scores = {}, {}
            for sid in test_ids:
                ref_pre = torch.from_numpy(ref_emb_pre[sid]).to(device)
                ref_fin = torch.from_numpy(ref_emb_fin[sid]).to(device)
                pre_scores[sid] = [cosine_similarity(torch.from_numpy(emb1_pre).to(device), ref_pre).item(),
                                   cosine_similarity(torch.from_numpy(emb2_pre).to(device), ref_pre).item()]
                fin_scores[sid] = [cosine_similarity(torch.from_numpy(emb1_fin).to(device), ref_fin).item(),
                                   cosine_similarity(torch.from_numpy(emb2_fin).to(device), ref_fin).item()]

            pred_id1_pre = max(pre_scores, key=lambda k: pre_scores[k][0])
            pred_id2_pre = max(pre_scores, key=lambda k: pre_scores[k][1])
            pred_id1_fin = max(fin_scores, key=lambda k: fin_scores[k][0])
            pred_id2_fin = max(fin_scores, key=lambda k: fin_scores[k][1])

            pre_correct = (pred_id1_pre == id1 and pred_id2_pre == id2) or (pred_id1_pre == id2 and pred_id2_pre == id1)
            fin_correct = (pred_id1_fin == id1 and pred_id2_fin == id2) or (pred_id1_fin == id2 and pred_id2_fin == id1)
            correct_pre += pre_correct
            correct_fin += fin_correct
            total += 1

    for metric in results:
        avg = np.mean(results[metric])
        print(f"Average {metric}: {avg:.2f}")
    rank1_pre = correct_pre / total * 100
    rank1_fin = correct_fin / total * 100
    print(f"Pre-trained WavLM Rank-1 Accuracy: {rank1_pre:.2f}%")
    print(f"Fine-tuned WavLM Rank-1 Accuracy: {rank1_fin:.2f}%")

# Extract embedding
def extract_embedding(waveform, model):
    inputs = feature_extractor(waveform.tolist(), sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(device)
    with torch.no_grad():
        outputs = model(input_values)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding

# Run pipeline
print("Training SepID-Enhance Pipeline...")
train_pipeline()
print("\nEvaluating on Test Set...")
evaluate_pipeline()

"""# **Question 2: MFCC Feature Extraction and Comparative Analysis of Indian Languages**

# Task A.
"""

#Download the audio dataset from Kaggle
import kagglehub

# Download latest version
dataset_root = kagglehub.dataset_download("hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages")

print("Path to dataset files:", dataset_root)

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Paths
dataset_root = "/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset/"

# Selected languages
languages = ["Hindi", "Tamil", "Bengali"]
samples_per_lang = 5

# Function to extract MFCCs
def extract_mfcc(audio_path, n_mfcc=13, hop_length=512, n_fft=2048):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    return mfcc, sr

# Function to plot MFCC spectrogram
def plot_mfcc(mfcc, sr, title, hop_length=512):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, hop_length=hop_length, x_axis="time", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.show()

# Collect and process samples
mfcc_data = {lang: [] for lang in languages}
for lang in languages:
    lang_path = os.path.join(dataset_root, lang)
    audio_files = [f for f in os.listdir(lang_path) if f.endswith(".mp3")]
    if not audio_files:
        print(f"No .wav files found in {lang_path}")
        continue

    # Limit to first few samples for visualization
    for i, audio_file in enumerate(tqdm(audio_files[:samples_per_lang], desc=f"Processing {lang}")):
        audio_path = os.path.join(lang_path, audio_file)
        mfcc, sr = extract_mfcc(audio_path)
        mfcc_data[lang].append(mfcc)

        # Plot MFCC spectrogram
        plot_mfcc(mfcc, sr, f"{lang} Sample {i+1} MFCC Spectrogram")

# Statistical analysis
def compute_stats(mfcc_list, lang):
    mfcc_flat = np.concatenate([m.T for m in mfcc_list], axis=0)  # Flatten time axis
    mean_mfcc = np.mean(mfcc_flat, axis=0)
    var_mfcc = np.var(mfcc_flat, axis=0)
    print(f"\n{lang} MFCC Statistics:")
    print(f"Mean MFCC (across coefficients): {mean_mfcc}")
    print(f"Variance MFCC (across coefficients): {var_mfcc}")
    return mean_mfcc, var_mfcc

# Perform statistical analysis
stats = {}
for lang in languages:
    if mfcc_data[lang]:
        stats[lang] = compute_stats(mfcc_data[lang], lang)

"""# TASK B"""

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt

# Paths
dataset_root = "/kaggle/input/audio-dataset-with-10-indian-languages/Language Detection Dataset/"

# All 10 languages
languages = ["Hindi", "Tamil", "Bengali", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi", "Urdu"]
lang_to_idx = {lang: idx for idx, lang in enumerate(languages)}

# Function to extract MFCCs
def extract_mfcc(audio_path, n_mfcc=13, hop_length=512, n_fft=2048):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    # Take mean across time axis to get fixed-size feature vector
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# Collect data
X = []  # Features (MFCCs)
y = []  # Labels (language indices)

for lang in languages:
    lang_path = os.path.join(dataset_root, lang)
    audio_files = [f for f in os.listdir(lang_path) if f.endswith(".mp3")]
    if not audio_files:
        print(f"No .wav files found in {lang_path}")
        continue

    for audio_file in tqdm(audio_files, desc=f"Processing {lang}"):
        audio_path = os.path.join(lang_path, audio_file)
        mfcc = extract_mfcc(audio_path)
        X.append(mfcc)
        y.append(lang_to_idx[lang])

X = np.array(X)  # Shape: (n_samples, n_mfcc)
y = np.array(y)  # Shape: (n_samples,)

print(f"Total samples: {X.shape[0]}, Features per sample: {X.shape[1]}")

# Preprocessing: Normalize features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=languages)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - Random Forest")
plt.tight_layout()
plt.show()