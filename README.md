# 🧠 Reasoning Trace Generator: Chain-of-Thought AI

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-Flan--T5-orange)
![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the source code for the **Reasoning Trace Generator**, a solution developed for the [**Ultimate AI & Tech Challenge**](https://www.kaggle.com/competitions/winners-will-will-exciting-prizes) on Kaggle.

The project focuses on teaching an AI model to "think step-by-step" by generating **Chain-of-Thought (CoT)** reasoning traces for mathematical problems, rather than just outputting the final answer.

---

## 📖 Overview

Standard language models often struggle with multi-step arithmetic or logic when forced to give an immediate answer. This project fine-tunes a Sequence-to-Sequence model to produce a structured "thought process" before arriving at a solution.

**Key Features:**
* **Fine-tuned LLM:** Uses Google's `flan-t5-small` as the base model.
* **Chain-of-Thought:** Generates explicit reasoning steps (e.g., "Step 1: Identify numbers...").
* **Hybrid Architecture:** Combines the neural model with a rule-based fallback system to ensure 100% accuracy on standard arithmetic patterns.
* **Generalization:** Tested against out-of-distribution (OOD) numbers and varied question phrasings.

---

## 📂 Dataset

The model was trained on the `AI_Thought_Chain_Dataset_1000.csv`, which consists of 1,000 examples of mathematical questions paired with structured reasoning traces.

| Input (Question) | Target (Reasoning Trace) |
| :--- | :--- |
| `What is 30 + 30?` | `Step 1: Identify the numbers.`<br>`Step 2: Add 30 + 30.`<br>`Step 3: The result is 60.` |

* **Train/Val Split:** 800 Training / 200 Validation
* **Preprocessing:** Inputs are prefixed with `generate reasoning: ` to prime the T5 task head.

---

## 🛠️ Technical Approach

### 1. Model Architecture
We utilized **Flan-T5-Small** (77M params), an instruction-tuned variant of T5. It was chosen for its efficiency and strong zero-shot capabilities in text-to-text tasks.

### 2. Training Configuration
* **Optimizer:** AdamW
* **Learning Rate:** 3e-4
* **Batch Size:** 16
* **Epochs:** 100 (with Early Stopping)
* **Max Token Length:** 64 (Input) / 128 (Target)

### 3. Hybrid Inference Strategy
To mitigate hallucinations typical in small LLMs (e.g., adding `1500 + 1500` incorrectly), a **Hybrid Generation Function** was implemented:
1.  **Model Attempt:** The neural network generates a reasoning trace.
2.  **Verification:** A regex-based rule engine attempts to solve the problem deterministically.
3.  **Consensus/Fallback:** If the model's numbers don't match the deterministic calculation, the system falls back to the rule-based output to guarantee correctness in the competition submission.

---

## 📊 Results & Performance

The model learned the structural pattern of the reasoning traces effectively.

* **Validation Loss:** Converged to ~0.02
* **Pattern Matching:** Successfully replicates the "Step 1... Step 2... Step 3..." format.
* **Generalization:** * *Standard Formatting:* High accuracy.
    * *Different Phrasings:* Robust to "Calculate...", "What's...", and "Add..." prompts.
    * *OOD Numbers:* The small model struggles with arithmetic on numbers larger than those seen in training (e.g., >1000), necessitating the hybrid fallback approach.

---

## 🚀 Installation & Usage

### Prerequisites
The code is designed to run in a Jupyter Notebook environment (Kaggle/Colab) or locally with GPU support.

```bash
pip install transformers datasets evaluate accelerate sentencepiece scikit-learn torch matplotlib seaborn
```
### Running the Code
1.  Clone this repository.
2.  Ensure the dataset is placed in the correct directory (or update the path in the notebook).
3.  Run `ultimate-ai-tech-challenge-chain-of-thoughts.ipynb`.

### Inference Example
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("./reasoning_model")
model = T5ForConditionalGeneration.from_pretrained("./reasoning_model")

question = "What is 42 + 42?"
input_text = f"generate reasoning: {question}"
inputs = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**inputs, max_length=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
## 🏆 Kaggle Competition

This project was created for the **Ultimate AI & Tech Challenge**.
* **Notebook Link:** [View on Kaggle](https://www.kaggle.com/code/basselashraf/ultimate-ai-tech-challenge-chain-of-thoughts)
* **Competition:** [Winners Will Win Exciting Prizes](https://www.kaggle.com/competitions/winners-will-will-exciting-prizes)

---

## 📜 License

This project is open-sourced under the MIT License.
