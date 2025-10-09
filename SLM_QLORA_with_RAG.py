# install libraries
!pip install -q transformers datasets peft accelerate bitsandbytes torch sentence-transformers faiss-cpu evaluate rouge_score bert_score trl fpdf huggingface_hub

print("✅ Libraries installed")

# Commented out IPython magic to ensure Python compatibility.


# Install necessary tools
!pip install --upgrade pip
!pip install -q torch transformers datasets accelerate peft trl sentence-transformers evaluate rouge_score bert_score faiss-cpu
!pip install -q git+https://github.com/EleutherAI/lm-evaluation-harness.git
print("✅ Tools installed. RESTART RUNTIME before running the rest of the script.") #I encountered multiple issue with the runtime, and had to install and restart it multiple times before continuing

# Bring in the tools
import os, torch, traceback, threading, time
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from google.colab import drive

#  Set up how we want to train
MODEL_ID = "microsoft/phi-2"
BATCH_SIZE = 1
GRAD_ACC = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
SAVE_STEPS = 200
MAX_SEQ_LEN = 1024
DRY_RUN = False       # Set this to True to quickly check if things work

# Where to save our project files
PROJECT_DIR = "/content/drive/MyDrive/SLM_Project_QLoRA"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "phi-2_fp16_squad_finetuned")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "phi2_fp16_checkpoints")
LOG_DIR = os.path.join(PROJECT_DIR, "phi2_fp16_logs")

# Make the folders if they don't exist
for d in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)

# Connect to Google drive
if os.path.exists("/content/drive"):
    !rm -rf /content/drive/*
drive.mount("/content/drive", force_remount=True)
print("✅ Google Drive connected")

# Check graphics card
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Get the model and its helper (tokenizer)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map="auto",
)
model.config.use_cache = False
# Prepare model for training
model = prepare_model_for_kbit_training(model)

# Set up LoRA (a way to train faster)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Get the data (SQuAD dataset)
max_retries = 5
for i in range(max_retries):
    try:
        dataset = load_dataset("squad", split="train")
        break
    except Exception as e:
        print(f"Try {i+1} failed: {e}")
        if i < max_retries - 1:
            time.sleep(10)
        else:
            raise

# Make the data look like what the model expects
def format_dataset(example):
    ans = example["answers"]["text"][0] if example["answers"]["text"] else ""
    return {"text": f"Context:\n{example['context']}\n\nQuestion:\n{example['question']}\n\nAnswer:\n{ans}"}

dataset = dataset.map(format_dataset, remove_columns=dataset.column_names)

# Turn text into numbers for the model
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LEN, padding="longest")

dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

# If doing a dry run, use only a small part of the data
if DRY_RUN:
    dataset = dataset.select(range(8))
    print("🧪 DRY RUN is on: using only 8 examples.")

# Set up training settings
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR, # Where to save model backups
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACC,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_dir=LOG_DIR,
    logging_steps=50,
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=2,
    fp16=True,
    bf16=False,
    report_to=["tensorboard"]
)

# Set up the trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args
)

# Start TensorBoard
def launch_tensorboard():
#     %load_ext tensorboard
#     %tensorboard --logdir {LOG_DIR}

threading.Thread(target=launch_tensorboard).start()
print("🚀 TensorBoard started in the background.")

#  Quick test run
if DRY_RUN:
    print("🧪 Doing a quick test run...")
    try:
        batch = next(iter(trainer.get_train_dataloader()))
        batch = {k: v.to(next(model.parameters()).device) for k,v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
        print("✅ Quick test run worked. Output shape:", out.logits.shape)
    except Exception:
        print("❌ Quick test run failed:")
        traceback.print_exc()
    raise SystemExit("✅ QUICK TEST DONE — change DRY_RUN=False to train for real.")

# Start training
last_ckpt = get_last_checkpoint(CHECKPOINT_DIR)
if last_ckpt:
    print(f"🔁 Continuing from saved point: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
else:
    print("🆕 Starting training from the beginning")
    trainer.train()

# Save the final trained model
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Final model saved at {OUTPUT_DIR}")

#  VECTOR DATABASE WITH FAISS

import os
import faiss
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import pickle

# Load your dataset (you can replace 'squad' with your own dataset)
dataset = load_dataset("squad", split="train")

# Prepare texts
texts = [
    f"Context:\n{ex['context']}\n\nQuestion:\n{ex['question']}\n\nAnswer:\n{ex['answers']['text'][0] if ex['answers']['text'] else ''}"
    for ex in dataset
]

# Use a SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("🔹 Generating embeddings...")
embeddings = embedding_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

# Create FAISS index
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings)
print(f"✅ FAISS index created with {index.ntotal} vectors.")

# Save index and texts for retrieval
faiss.write_index(index, "squad_faiss_index.idx")
with open("squad_texts.pkl", "wb") as f:
    pickle.dump(texts, f)
print("✅ Vector database saved (FAISS + texts).")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def rag_query(query, top_k=3, max_length=128):
    try:
        # embed query
        embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        q_emb = embed_model.encode([query], convert_to_numpy=True)
        # retrieve top-k
        D, I = index.search(q_emb, top_k)
        retrieved_texts = [texts[i] for i in I[0]]
        # prepare input
        input_text = "\n\n".join(retrieved_texts) + f"\n\nQuestion: {query}\nAnswer:"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        # generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_length)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# test RAG
example_queries = [
    "Who is the main character in SQuAD?",
    "What is the context of the first question?",
    "Give a sample answer for a SQuAD question."
]

for q in example_queries:
    print("Q:", q)
    print("A:", rag_query(q), "\n")

import evaluate

# prepare sample predictions and references
sample_dataset = load_dataset("squad", split="train[:50]")  # small for testing
predictions = [rag_query(f"{ex['context']}\nQuestion: {ex['question']}") for ex in sample_dataset]
references = [ex['answers']['text'][0] for ex in sample_dataset]

# compute ROUGE
rouge = evaluate.load("rouge")
rouge_res = rouge.compute(predictions=predictions, references=references)
print("ROUGE:", rouge_res)

# compute BERTScore
bertscore = evaluate.load("bertscore")
bertscore_res = bertscore.compute(predictions=predictions, references=references, lang="en")
print("BERTScore:", bertscore_res)

# ============================================================
# STEP 3: ASK RAG QUESTIONS NOT IN SQuAD
# ============================================================

# Define some questions that are not directly from the SQuAD dataset
new_queries = [
    "What is the capital of France?",
    "How does photosynthesis work?",
    "Tell me about the history of the internet."
]

print("Asking RAG questions not in SQuAD:")
for query in new_queries:
    print(f"\nQuestion: {query}")
    response = rag_query(query)
    print(f"Answer: {response}")

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", 'B', 16)
pdf.cell(0, 10, "SLM Fine-Tuning Report", ln=True, align='C')
pdf.set_font("Arial", '', 12)
pdf.ln(5)

# Model info
pdf.cell(0, 10, f"Model: {MODEL_ID}", ln=True)
pdf.cell(0, 10, f"Fine-tuned model path: {OUTPUT_DIR}", ln=True)
pdf.cell(0, 10, f"Batch size: {BATCH_SIZE}, Grad Acc: {GRAD_ACC}", ln=True)
pdf.cell(0, 10, f"Learning rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}", ln=True)
pdf.ln(5)

# RAG example queries
pdf.cell(0, 10, "RAG Example Queries:", ln=True)
for q in example_queries:
    ans = rag_query(q)
    pdf.multi_cell(0, 8, f"Q: {q}\nA: {ans}\n")
    pdf.ln(2)

# Evaluation metrics
pdf.cell(0, 10, "Evaluation Metrics (sample 50 items):", ln=True)
pdf.multi_cell(0, 8, f"ROUGE:\n{rouge_res}\n")
pdf.multi_cell(0, 8, f"BERTScore:\n{bertscore_res}\n")

# Save PDF
pdf_path = os.path.join(PROJECT_DIR, "SLM_finetune_full_report.pdf")
pdf.output(pdf_path)
print("✅ PDF report saved at:", pdf_path)

import pandas as pd

rouge_data = {key: [value] for key, value in rouge_res.items()}


bertscore_data = {}
for metric in ['precision', 'recall', 'f1']:
    if metric in bertscore_res and isinstance(bertscore_res[metric], list):
        bertscore_data[metric] = [sum(bertscore_res[metric]) / len(bertscore_res[metric])]
    else:
        bertscore_data[metric] = [None] # Handle cases where a metric might be missing or not a list

# Create DataFrames
rouge_df = pd.DataFrame(rouge_data)
bertscore_df = pd.DataFrame(bertscore_data)

# Combine the results into a single table
benchmark_df = pd.concat([rouge_df, bertscore_df], axis=1)

print("📊 Model Performance Benchmark:")
display(benchmark_df)
