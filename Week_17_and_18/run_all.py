"""
Master Execution Script for Week 17 & Week 18 PBL Tasks
Runs all 8 Deep Learning & NLP modules sequentially with clean console formatting.
"""

import sys
import subprocess
from pathlib import Path

MODULES = [
    ("Part 1A: Tensors & Autograd", "01_pytorch_basics/tensors_and_autograd.py"),
    ("Part 1B: Canonical Training Loop", "01_pytorch_basics/training_loop.py"),
    ("Part 2:  Convolutional Neural Networks (CNNs & Predictions)", "02_cnns/cnn_image_classification.py"),
    ("Part 3:  Sequence Models (RNN vs LSTM vs GRU)", "03_sequence_models/sequence_modeling.py"),
    ("Part 4:  Regularization (Dropout, BatchNorm, Early Stopping)", "04_regularization/regularization_techniques.py"),
    ("Part 5:  Transfer Learning with Pretrained CNNs", "05_transfer_learning/transfer_learning.py"),
    ("Part 6:  Hugging Face Transformers NLP", "06_transformers/transformers_nlp.py"),
    ("Part 7:  LLM APIs & Open Models (OpenAI, Anthropic, Ollama)", "07_llm_apis/llm_apis_demo.py"),
]

def main():
    root_dir = Path(__file__).resolve().parent
    print("=" * 72, flush=True)
    print("  WEEK 17 & 18: DEEP LEARNING & LLM CURRICULUM TEST RUNNER", flush=True)
    print("=" * 72, flush=True)

    for idx, (title, rel_path) in enumerate(MODULES, 1):
        script_path = root_dir / rel_path
        print(f"\n[{idx}/{len(MODULES)}] EXECUTING: {title}", flush=True)
        print(f"Path: {script_path}", flush=True)
        print("-" * 72, flush=True)

        cmd = [sys.executable, "-u", str(script_path)]
        result = subprocess.run(cmd, text=True, capture_output=True)

        # Print child output cleanly without terminal buffer tearing
        if result.stdout:
            sys.stdout.write(result.stdout)
            sys.stdout.flush()
        if result.stderr:
            sys.stderr.write(result.stderr)
            sys.stderr.flush()

        if result.returncode != 0:
            print(f"\n[ERROR] Module '{title}' exited with code {result.returncode}", flush=True)
        else:
            print(f"\n-> [{title}] finished successfully.", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("  ALL MODULES EXECUTED SUCCESSFULLY!", flush=True)
    print("=" * 72, flush=True)

if __name__ == "__main__":
    main()
