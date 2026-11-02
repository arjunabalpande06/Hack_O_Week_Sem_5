# PBL Hack-O-Week: Week 17 & Week 18
## Deep Learning Foundations, Architectures, Transfer Learning & LLMs

A production-grade, modular, and self-contained laboratory curriculum covering fundamental and advanced Deep Learning concepts in PyTorch, Hugging Face Transformers, and Large Language Model APIs.

---

## 📁 Repository Structure

```text
Week 17 & Week 18/
├── 01_pytorch_basics/
│   ├── tensors_and_autograd.py       # Tensors, ops, CUDA, reverse-mode Autograd DAG
│   └── training_loop.py              # Canonical train-eval loop (zero_grad, backward, step)
├── 02_cnns/
│   └── cnn_image_classification.py   # Conv2d, MaxPool2d, Feature Maps, Image Classifier
├── 03_sequence_models/
│   └── sequence_modeling.py          # Vanilla RNN vs LSTM vs GRU comparison & gating
├── 04_regularization/
│   └── regularization_techniques.py  # Dropout, BatchNorm, EarlyStopping callback
├── 05_transfer_learning/
│   └── transfer_learning.py          # Pretrained ResNet-18 (Feature Extraction & Fine-Tuning)
├── 06_transformers/
│   └── transformers_nlp.py           # HF Pipelines, Tokenizers, Scaled Dot-Product Attention
├── 07_llm_apis/
│   └── llm_apis_demo.py              # OpenAI, Anthropic, & Ollama (Generation, Summarization, QA)
├── run_all.py                        # Master script running all 7 modules sequentially
├── DATASETS_GUIDE.md                 # Complete assessment on Kaggle datasets
└── requirements.txt                  # Python dependencies
```

---

## ❓ Is Any Dataset Required from Kaggle?

> **SHORT ANSWER: NO.**
> 
> You **do NOT need to download any dataset from Kaggle** to complete or execute these tasks.
> 
> - **Self-Contained Data Generators**: All modules feature built-in programmatic datasets (synthetic regression surfaces, 28×28 geometric image patterns, multi-frequency sinusoidal sequential waves, and noisy feature splits).
> - **Zero-Setup Benchmarking**: You can clone and run every single script immediately without waiting for multi-gigabyte Kaggle downloads or setting up Kaggle API keys.
> - **Pluggable Architecture**: If your course instructor specifically asks for a Kaggle dataset (e.g. *Dogs vs. Cats* or *IMDB 50k Reviews*), our codebase provides pre-configured `ImageFolder` and DataFrame loader snippets documented in [DATASETS_GUIDE.md](file:///d:/Symbiosis/SEM%205/PBL%20Hack%20O%20Week/Week%2017%20&%20Week%2018/DATASETS_GUIDE.md).

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Execute the Entire Suite
To run all 7 modules in sequence with full performance logging:
```powershell
python run_all.py
```

### 3. Run Individual Modules
```powershell
# Part 1: Tensors, Autograd & Training Loop
python 01_pytorch_basics\tensors_and_autograd.py
python 01_pytorch_basics\training_loop.py

# Part 2: Convolutional Neural Networks
python 02_cnns\cnn_image_classification.py

# Part 3: Sequence Models (RNN vs LSTM vs GRU)
python 03_sequence_models\sequence_modeling.py

# Part 4: Regularization (Dropout, BatchNorm, Early Stopping)
python 04_regularization\regularization_techniques.py

# Part 5: Transfer Learning with ResNet-18
python 05_transfer_learning\transfer_learning.py

# Part 6: Hugging Face Transformers
python 06_transformers\transformers_nlp.py

# Part 7: Large Language Model APIs
python 07_llm_apis\llm_apis_demo.py
```

---

## 📚 Curriculum Breakdown & Key Concepts

### Part 1: PyTorch Foundations & Training Loop
- **Tensors**: N-dimensional array allocation, device migration (`cpu` ↔ `cuda`), slicing, broadcasting, and matrix multiplication (`@`).
- **Autograd Engine**: How PyTorch constructs a dynamic Directed Acyclic Graph (DAG) during the forward pass. Demonstrates `requires_grad=True`, `.backward()`, gradient accumulation, and `.grad.zero_()`.
- **Training Loop**: The canonical 5-step PyTorch training recipe:
  1. `model.train()`
  2. `optimizer.zero_grad()`
  3. `loss = criterion(model(x), y)`
  4. `loss.backward()`
  5. `optimizer.step()`
  6. Evaluation via `model.eval()` with `with torch.no_grad():`.

### Part 2: Convolutional Neural Networks (CNNs)
- **Convolution (`nn.Conv2d`)**: Feature extraction using learnable kernels across spatial dimensions with stride and padding.
- **Pooling (`nn.MaxPool2d`)**: Spatial downsampling providing translation invariance and computational efficiency.
- **Classification Pipeline**: Stacking Conv-ReLU-Pool blocks feeding into a flattened fully-connected dense classification head.

### Part 3: Sequence Modeling (RNN vs. LSTM vs. GRU)
- **Vanilla RNN**: Simple recurrence $h_t = \tanh(W x_t + U h_{t-1} + b)$. Suffers from vanishing/exploding gradients on long sequences.
- **LSTM (Long Short-Term Memory)**: Overcomes vanishing gradients via an additive Cell State highway ($c_t$) controlled by 3 gates:
  - *Forget Gate* ($f_t$): What to discard from memory.
  - *Input Gate* ($i_t$): What new information to store.
  - *Output Gate* ($o_t$): What part of the cell state to output to $h_t$.
- **GRU (Gated Recurrent Unit)**: Lightweight alternative with 2 gates (*Reset* and *Update*) merging cell and hidden states.

### Part 4: Deep Learning Regularization
- **Dropout**: Randomly deactivates neuron activations with probability $p$ during training, preventing feature co-adaptation.
- **Batch Normalization**: Normalizes mini-batch distributions ($\mu=0, \sigma^2=1$) with learnable scale $\gamma$ and shift $\beta$, smoothing the optimization landscape.
- **Early Stopping**: Custom callback monitoring validation loss over a patience window, terminating training before overfitting sets in and rolling back to the optimal checkpoint.

### Part 5: Transfer Learning with Pretrained CNNs
- **Feature Extractor**: Freezing all pretrained convolutional backbone parameters (`param.requires_grad = False`) and replacing only the final linear head (`model.fc`) to classify custom classes.
- **Fine-Tuning**: Selectively unfreezing top layers (e.g. `model.layer4`) and applying differential learning rates ($10^{-4}$ for backbone, $10^{-3}$ for classification head).

### Part 6: Hugging Face Transformers
- **High-Level Pipelines**: Single-line inferences for `sentiment-analysis`, `summarization`, and `question-answering`.
- **AutoTokenizer & AutoModel**: Subword tokenization (WordPiece/BPE), input IDs, attention masks, and logit post-processing.
- **Scaled Dot-Product Attention**: Mathematical mechanics of:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Part 7: Large Language Model (LLM) APIs & Open Models
- **Commercial APIs**: OpenAI (`gpt-4o-mini`) and Anthropic Claude (`claude-3-5-sonnet`) with proper system prompting and parameter tuning.
- **Local Open Models**: Ollama integration (`llama3`, `mistral`) running entirely on local hardware with zero subscription fees.
- **Core Tasks**: Automated text generation, research paper abstract summarization, and context-grounded question answering.
