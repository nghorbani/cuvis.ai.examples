# EfficientAD — Training, Inference, and Architectures

**Source**: Adapted from “EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies.”  

---

## Algorithm 1: Training EfficientAD-S

Train a **student** network to (1) mimic a **teacher** feature extractor on normal images with *hard feature mining* and a *pretraining penalty*, and (2) predict the **autoencoder (AE)** reconstructions of teacher features. Train an **autoencoder** to reconstruct teacher features (to capture global logical constraints). The **teacher** remains frozen.

### Inputs
- **Teacher** `T: R^{3×256×256} → R^{384×64×64}`  
  Pretrained/frozen PDN (distilled from a large backbone).
- **Training images** `I_train` (normal-only), each `I ∈ R^{3×256×256}`  
- **Validation images** `I_val` (subset of normal images)  
- **ImageNet** (or other teacher-pretraining set) for the student’s pretraining penalty term.

### Initialize
1. **Student** `S: R^{3×256×256} → R^{768×64×64}`  
   (First 384 channels for teacher-matching, last 384 for AE-matching.)
2. **Autoencoder** `A: R^{3×256×256} → R^{384×64×64}`
3. **Teacher channel normalization (μ, σ)**  
   - For each channel, compute mean and std across teacher outputs on training images.
4. **Optimizer**  
   Adam with `lr = 1e-4`, `weight_decay = 1e-5` on `S` and `A`.

### Training Loop (70,000 iterations)
1. **Student–Teacher branch (local features)**  
   - Normalize teacher outputs.  
   - Student predicts same features (`Y^{ST}`).  
   - Loss = *hard feature loss* (top 0.1% highest errors) + *pretraining penalty* (ImageNet regularization).
2. **Autoencoder–Student branch (global features)**  
   - Augment input (`brightness`, `contrast`, `saturation`).  
   - Autoencoder reconstructs teacher features (`Y^A`).  
   - Student predicts AE reconstructions (`Y^{STAE}`).  
   - Losses: `L_AE` (teacher–AE diff), `L_STAE` (AE–student diff).
3. **Total loss**  
   - `L_total = L_ST + L_AE + L_STAE`  
   - Update `S` and `A`.  
   - After 66.5k iterations, reduce `lr = 1e-5`.

### Validation (quantile normalization)
- Compute local (`M^{ST}`) and global (`M^{AE}`) anomaly maps for validation set.  
- Extract quantiles:
  - Local: `(q_a^{ST}, q_b^{ST}) = (0.9, 0.995)` quantiles.  
  - Global: `(q_a^{AE}, q_b^{AE}) = (0.9, 0.995)` quantiles.  
- These define linear scaling to map `q_a → 0` and `q_b → 0.1`.

### Return
- Trained student `S` and autoencoder `A`  
- Frozen teacher `T`  
- Teacher normalization `(μ, σ)`  
- Quantile sets for both local and global maps  

---

## Algorithm 2: Inference with EfficientAD

Use the frozen **teacher**, trained **student**, and trained **autoencoder** to generate anomaly maps and image-level anomaly scores.

### Inputs
- **Test image** `I_test ∈ R^{3×256×256}`  
- **Trained models and parameters:**  
  - Teacher `T`, Student `S`, Autoencoder `A`  
  - Teacher normalization `(μ, σ)`  
  - Quantiles `(q_a^{ST}, q_b^{ST}, q_a^{AE}, q_b^{AE})`

### Steps
1. Forward pass:
   - `Y' = T(I_test)`  
   - `Y^S = S(I_test)` → split into `Y^{ST}` (first 384) and `Y^{STAE}` (last 384)  
   - `Y^A = A(I_test)`  
   - Normalize teacher: `Ŷ_c = (Y'_c − μ_c)/σ_c`
2. Compute squared diffs:
   - Local: `D^{ST} = (Ŷ − Y^{ST})^2`  
   - Global: `D^{STAE} = (Y^A − Y^{STAE})^2`
3. Aggregate into anomaly maps:
   - `M^{ST} = mean_c(D^{ST})`  
   - `M^{AE} = mean_c(D^{STAE})`  
   - Resize both to `256×256`
4. Normalize anomaly maps:
   - `M̂^{ST} = 0.1 * (M^{ST} − q_a^{ST}) / (q_b^{ST} − q_a^{ST})`  
   - `M̂^{AE} = 0.1 * (M^{AE} − q_a^{AE}) / (q_b^{AE} − q_a^{AE})`
5. Combine:
   - `M = 0.5 * M̂^{ST} + 0.5 * M̂^{AE}`
6. Image-level score:
   - `m_image = max(M)` (max pixel score)

### Outputs
- **Combined anomaly map** `M`  
- **Image-level anomaly score** `m_image`

---

## PDN Teacher Architectures

### Table 6 — Teacher Network (EfficientAD-S)

| Layer    | Stride | Kernel Size | Kernels | Padding | Activation |
|----------|--------|-------------|---------|---------|------------|
| Conv-1   | 1×1    | 4×4         | 128     | 3       | ReLU       |
| AvgPool-1| 2×2    | 2×2         | 128     | 1       | –          |
| Conv-2   | 1×1    | 4×4         | 256     | 3       | ReLU       |
| AvgPool-2| 2×2    | 2×2         | 256     | 1       | –          |
| Conv-3   | 1×1    | 3×3         | 256     | 1       | ReLU       |
| Conv-4   | 1×1    | 4×4         | 384     | 0       | –          |

> Student has the same architecture but **768 kernels** in Conv-4.

---

### Table 7 — Teacher Network (EfficientAD-M)

| Layer    | Stride | Kernel Size | Kernels | Padding | Activation |
|----------|--------|-------------|---------|---------|------------|
| Conv-1   | 1×1    | 4×4         | 256     | 3       | ReLU       |
| AvgPool-1| 2×2    | 2×2         | 256     | 1       | –          |
| Conv-2   | 1×1    | 4×4         | 512     | 3       | ReLU       |
| AvgPool-2| 2×2    | 2×2         | 512     | 1       | –          |
| Conv-3   | 1×1    | 1×1         | 512     | 0       | ReLU       |
| Conv-4   | 1×1    | 3×3         | 512     | 1       | ReLU       |
| Conv-5   | 1×1    | 4×4         | 384     | 0       | ReLU       |
| Conv-6   | 1×1    | 1×1         | 384     | 0       | –          |

> Student has the same architecture but **768 kernels** in Conv-5 and Conv-6.

---

### Table 8 — Autoencoder (EfficientAD-S & M)

| Layer     | Stride | Kernel Size | Kernels | Padding | Activation |
|-----------|--------|-------------|---------|---------|------------|
| EncConv-1 | 2×2    | 4×4         | 32      | 1       | ReLU       |
| EncConv-2 | 2×2    | 4×4         | 32      | 1       | ReLU       |
| EncConv-3 | 2×2    | 4×4         | 64      | 1       | ReLU       |
| EncConv-4 | 2×2    | 4×4         | 64      | 1       | ReLU       |
| EncConv-5 | 2×2    | 4×4         | 64      | 1       | ReLU       |
| EncConv-6 | 1×1    | 8×8         | 64      | 0       | –          |
| Bilinear-1| –      | –           | –       | –       | Resize to 3×3 |
| DecConv-1 | 1×1    | 4×4         | 64      | 2       | ReLU       |
| Dropout-1 | –      | –           | –       | –       | Rate=0.2   |
| Bilinear-2| –      | –           | –       | –       | Resize to 8×8 |
| DecConv-2 | 1×1    | 4×4         | 64      | 2       | ReLU       |
| Dropout-2 | –      | –           | –       | –       | Rate=0.2   |
| Bilinear-3| –      | –           | –       | –       | Resize to 15×15 |
| DecConv-3 | 1×1    | 4×4         | 64      | 2       | ReLU       |
| Dropout-3 | –      | –           | –       | –       | Rate=0.2   |
| Bilinear-4| –      | –           | –       | –       | Resize to 32×32 |
| DecConv-4 | 1×1    | 4×4         | 64      | 2       | ReLU       |
| Dropout-4 | –      | –           | –       | –       | Rate=0.2   |
| Bilinear-5| –      | –           | –       | –       | Resize to 63×63 |
| DecConv-5 | 1×1    | 4×4         | 64      | 2       | ReLU       |
| Dropout-5 | –      | –           | –       | –       | Rate=0.2   |
| Bilinear-6| –      | –           | –       | –       | Resize to 127×127 |
| DecConv-6 | 1×1    | 4×4         | 64      | 2       | ReLU       |
| Dropout-6 | –      | –           | –       | –       | Rate=0.2   |
| Bilinear-7| –      | –           | –       | –       | Resize to 64×64 |
| DecConv-7 | 1×1    | 3×3         | 64      | 1       | ReLU       |
| DecConv-8 | 1×1    | 3×3         | 384     | 1       | –          |

---

## Comments on Algorithm 1 & 2

- **Initialization**:  
  Teacher, student, and autoencoder are initialized with PyTorch default initialization.  
- **Augmentation**:  
  Student–teacher branch trains without augmentation; autoencoder branch trains with brightness/contrast/saturation augmentation.  
- **Inference**:  
  Only one forward pass per image; no augmentation.  
- **Image sizes**:  
  Input images are resized to `256×256`. Anomaly maps are resized back to the original resolution.  
- **Batch size**:  
  1 for training and inference.  
- **Student output split**:  
  - Channels `[:384]` → mimic teacher  
  - Channels `[384:]` → mimic AE  
- **Quantile normalization**:  
  Ensures local and global anomaly maps are on similar scales before averaging.  
- **Normalization**:  
  Uses torchvision ImageNet normalization (mean 0.485/0.456/0.406, std 0.229/0.224/0.225).  

---
