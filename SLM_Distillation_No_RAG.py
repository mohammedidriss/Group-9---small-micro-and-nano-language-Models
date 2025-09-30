# -*- coding: utf-8 -*-

# Get the stuff we need - install libraries
!pip install -q transformers datasets peft accelerate bitsandbytes trl torch

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from datasets import load_dataset
import torch.nn.functional as F
import torch.nn as nn
import os

# Done with setup!
print("Libraries installed and imported successfully.")



#configuration- Set all the knobs and dials
class DistillationConfig:
    # Model IDs - big one and small one
    TEACHER_MODEL_ID = "microsoft/phi-2"
    STUDENT_MODEL_OUTPUT_DIR = "./distilled_student_model"

    # Data stuff
    DATASET_ID = "databricks/databricks-dolly-15k"
    DATASET_SUBSET_SIZE = 3000
    MAX_TOKEN_LENGTH = 512

    # Distillation magic numbers
    ALPHA = 0.5 # Balances the two losses
    TEMPERATURE = 2.0 # Softens predictions

    # Training junk
    NUM_TRAIN_EPOCHS = 1
    BATCH_SIZE = 4 # Keep it tiny for this GPU
    LEARNING_RATE = 5e-5
    OUTPUT_DIR = "./training_output"

config = DistillationConfig()

# loading data- Load the big model (Teacher) and make a tiny one (Student)
print("Loading tokenizer and teacher model...")

# Get the word-to-number
tokenizer = AutoTokenizer.from_pretrained(config.TEACHER_MODEL_ID, trust_remote_code=True)
# Make sure it knows how to pad sentences
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Config for loading the big model in 4-bit (saves memory!)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

teacher_model = AutoModelForCausalLM.from_pretrained(
    config.TEACHER_MODEL_ID,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto" # Put it wherever
)
# training
teacher_model.eval()
print("Teacher model loaded successfully in 4-bit.")

# Make the tiny model config (Student)
student_config = GPT2Config(
    vocab_size=len(tokenizer),
    n_layer=6,            # Less layers than Teacher!
    n_head=12,            # Less heads
    n_embd=768,           # Smaller brain size
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

# Create the tiny model
student_model = GPT2LMHeadModel(student_config)
# Send Student to the GPU
student_model = student_model.to(teacher_model.device)

print(f"Student model created with {student_model.num_parameters():,} parameters.")
print(f"Teacher model has {teacher_model.num_parameters():,} parameters (in 4-bit).")



# prepare the dataset - Get the data ready for munching
print("Preparing the dataset...")


dataset = load_dataset(config.DATASET_ID, split='train')

dataset = dataset.select(range(config.DATASET_SUBSET_SIZE))


def tokenize_function(examples):
    formatted_texts = []
    for i in range(len(examples["instruction"])):
        text = f"Instruction:\n{examples['instruction'][i]}\n\nResponse:\n{examples['response'][i]}"
        formatted_texts.append(text)


    return tokenizer(
        formatted_texts,
        padding="max_length",
        truncation=True,
        max_length=config.MAX_TOKEN_LENGTH,
    )


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names # Get rid of the old text
)


print("Dataset prepared and tokenized.")


# The special task for distillation
class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.loss_fct = nn.KLDivLoss(reduction="none")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Get student's output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        student_logits = outputs_student.logits

        # Get teacher's output (no training for Teacher)
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            teacher_logits = outputs_teacher.logits

        # Make vocab sizes match for math
        student_vocab_size = student_logits.size(-1)
        teacher_vocab_size = teacher_logits.size(-1)

        if student_vocab_size != teacher_vocab_size:
            padding_size = student_vocab_size - teacher_vocab_size
            teacher_logits = F.pad(teacher_logits, (0, padding_size), "constant", -1e9) # Pad with super small number

        # Calculate KL loss on tokens that aren't padding
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand_as(student_logits)
        else:
            mask = torch.ones_like(student_logits)

        # Soften predictions with Temperature
        soft_student_logits = F.log_softmax(student_logits / config.TEMPERATURE, dim=-1)
        soft_teacher_logits = F.softmax(teacher_logits / config.TEMPERATURE, dim=-1)

        # Calculate loss per token
        kl_loss_per_token = self.loss_fct(soft_student_logits, soft_teacher_logits)

        # Mask out padding and sum up
        masked_kl_loss = (kl_loss_per_token * mask).sum()
        num_active_tokens = mask.sum()

        # Normalize by number of real tokens
        loss_kl = masked_kl_loss / num_active_tokens

        # Scale KL loss by T^2
        loss_kl = loss_kl * (config.TEMPERATURE ** 2)

        # Mix the two losses together
        loss = (1 - config.ALPHA) * student_loss + config.ALPHA * loss_kl

        return (loss, outputs_student) if return_outputs else loss


=================================================
#  training commence!
print("Starting training...")

# Set up the training rules
training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=500,
    report_to="none",
    fp16=True,
)

# Get the special trainer
distiller = DistillationTrainer(
    model=student_model,
    teacher_model=teacher_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Train that model!
distiller.train()

print("Training finished.")

# See if the little guy learned anything
print("Saving the final student model...")

# Save the student and its tokenizer
unwrapped_model = distiller.model.module if hasattr(distiller.model, 'module') else distiller.model
unwrapped_model.save_pretrained(config.STUDENT_MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(config.STUDENT_MODEL_OUTPUT_DIR)

print(f"Model saved to {config.STUDENT_MODEL_OUTPUT_DIR}")


# Inference Function
def generate_response(prompt, model, tokenizer, max_length=150):
    """
    Generates a response from the model given a prompt. # Still keeping this helpful one
    """
    # Format the question
    formatted_prompt = f"Instruction:\n{prompt}\n\nResponse:\n"

    # Turn question into numbers and send to GPU
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Get the answer from the model
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True, # Add some randomness
        eos_token_id=tokenizer.eos_token_id # Stop when it's done
    )


    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response_text.split("Response:\n")[1].strip()

# test prompts for testing answers
print("\n--- Testing the distilled model ---")

# Load the saved model to test (just in case)
final_model = AutoModelForCausalLM.from_pretrained(config.STUDENT_MODEL_OUTPUT_DIR)
final_model.to(teacher_model.device) # Send to GPU

# test questions
prompt1 = "What is knowledge distillation in the context of deep learning?"
print(f"\nPrompt 1: {prompt1}")
response1 = generate_response(prompt1, final_model, tokenizer)
print(f"Generated Response:\n{response1}")

prompt2 = "Write a short, four-line poem about the stars."
print(f"\nPrompt 2: {prompt2}")
response2 = generate_response(prompt2, final_model, tokenizer)
print(f"Generated Response:\n{response2}")

prompt3 = "Brainstorm three potential names for a new brand of coffee."
print(f"\nPrompt 3: {prompt3}")
response3 = generate_response(prompt3, final_model, tokenizer)
print(f"Generated Response:\n{response3}")

print("\n--- Script execution complete ---")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the path to your saved model
STUDENT_MODEL_OUTPUT_DIR = "./distilled_student_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading the distilled model and tokenizer...")

try:
    # Load the tokenizer and model from the saved directory
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_OUTPUT_DIR)
    model.to(DEVICE)
    print("Model and tokenizer loaded successfully.")
except OSError:
    print(f"Error: Could not find a saved model at '{STUDENT_MODEL_OUTPUT_DIR}'.")
    print("Please make sure you have run the main training script successfully before running this cell.")
    exit()

# --- Inference Function (copied from the training script for convenience) ---
def generate_response(prompt, model, tokenizer, max_length=150):
    """
    Generates a response from the model given a prompt.
    """
    # Format the prompt into the template the model was trained on
    formatted_prompt = f"Instruction:\n{prompt}\n\nResponse:\n"

    # Tokenize the input and move it to the GPU/CPU
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate a response from the model
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the generated tokens back into text
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the generated response part
    # We split by "Response:\n" and take the second part.
    try:
        return response_text.split("Response:\n")[1].strip()
    except IndexError:
        return response_text # Fallback if the template isn't perfectly followed

# --- List of new, diverse prompts to test the model ---
new_prompts = [
    "Explain the plot of the movie 'Inception' in three sentences.",
    "Write a Python function that takes a list of numbers and returns the sum.",
    "Continue the following story: The old lighthouse stood on the cliff's edge, its light having gone out for the first time in a century. Suddenly, a strange green glow emanated from the rocks below...",
    "What are the main differences between a cat and a dog?",
    "Provide a simple recipe for making pancakes."
]


print("\n--- Running additional inference tests ---")

# Loop through the new prompts and generate a response for each one
for i, prompt in enumerate(new_prompts):
    print(f"\n--- Prompt {i+1} ---")
    print(f"Instruction: {prompt}")
    response = generate_response(prompt, model, tokenizer)
    print(f"Generated Response:\n{response}")

print("\n--- Additional testing complete ---")

# =============================================================================
# SECTION 1: SETUP AND DEPENDENCIES
# =============================================================================
# This section installs all required evaluation libraries, including bert_score.

!pip install -q evaluate rouge_score accelerate bert_score

import torch
import time
import evaluate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import numpy as np

# =============================================================================
# SECTION 2: LOAD MODELS AND EVALUATION DATASET
# =============================================================================
# We load both models, the tokenizer, and an evaluation dataset.

# --- Configuration ---
STUDENT_MODEL_DIR = "./distilled_student_model"
TEACHER_MODEL_ID = "microsoft/phi-2"
EVAL_DATASET_ID = "databricks/databricks-dolly-15k"
NUM_EVAL_SAMPLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading models and tokenizer for evaluation...")

# --- Load Models (Student and Teacher) ---
try:
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_DIR)
    student_model = AutoModelForCausalLM.from_pretrained(STUDENT_MODEL_DIR).to(DEVICE)
    print("✅ Distilled student model loaded.")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, quantization_config=bnb_config, trust_remote_code=True, device_map="auto"
    )
    print("✅ Original teacher model (phi-2) loaded.")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# --- Load Evaluation Dataset ---
eval_dataset = load_dataset(EVAL_DATASET_ID, split=f'train[{3000}:{3000 + NUM_EVAL_SAMPLES}]')
eval_prompts = [item['instruction'] for item in eval_dataset]
print(f"✅ Loaded {len(eval_prompts)} samples for evaluation.")

# =============================================================================
# SECTION 3: GENERATE RESPONSES FOR EVALUATION
# =============================================================================
# We generate responses from both models once and reuse them for all metrics.

print("\n--- Generating responses for evaluation... ---")
student_generations = []
teacher_generations = [] # These will be the "references"

for prompt in eval_prompts:
    formatted_prompt = f"Instruction:\n{prompt}\n\nResponse:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)

    student_output = student_model.generate(**inputs, max_new_tokens=100)
    student_text = tokenizer.decode(student_output[0], skip_special_tokens=True).split("Response:\n")[1].strip()
    student_generations.append(student_text)

    teacher_output = teacher_model.generate(**inputs, max_new_tokens=100)
    teacher_text = tokenizer.decode(teacher_output[0], skip_special_tokens=True).split("Response:\n")[1].strip()
    teacher_generations.append(teacher_text)

print("✅ Responses generated.")

# =============================================================================
# SECTION 4: CALCULATE ALL METRICS
# =============================================================================

# --- 1. Lexical Metrics (ROUGE & BLEU) ---
print("\n--- Calculating ROUGE and BLEU Scores ---")
rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

rouge_scores = rouge.compute(predictions=student_generations, references=teacher_generations)
bleu_scores = bleu.compute(predictions=student_generations, references=teacher_generations)
print("✅ ROUGE and BLEU scores computed.")

# --- 2. Semantic Metric (BERTScore) ---
print("\n--- Calculating BERTScore (this may take a moment)... ---")
bertscore = evaluate.load("bertscore")
bert_scores = bertscore.compute(predictions=student_generations, references=teacher_generations, lang="en")
# We take the average F1 score as the primary metric
avg_bert_score_f1 = np.mean(bert_scores['f1'])
print("✅ BERTScore computed.")

# --- 3. Intrinsic Metric (Perplexity) ---
print("\n--- Calculating Perplexity ---")
perplexity = evaluate.load("perplexity", module_type="metric")
student_ppl = perplexity.compute(model_id=STUDENT_MODEL_DIR, add_start_token=False, predictions=teacher_generations)
try:
    teacher_ppl = perplexity.compute(model_id=TEACHER_MODEL_ID, add_start_token=False, predictions=teacher_generations)
    print("✅ Perplexity computed for both models.")
except Exception as e:
    print(f"⚠️ Could not compute teacher perplexity (often due to quantization): {e}")
    teacher_ppl = {"mean_perplexity": "N/A"}

# --- 4. Efficiency Metrics (Speed & Size) ---
print("\n--- Calculating Inference Speed and Model Size ---")
student_times = []
teacher_times = []
inputs = tokenizer(eval_prompts[0], return_tensors="pt").to(DEVICE) # Use one prompt for timing
_ = student_model.generate(**inputs, max_new_tokens=2); _ = teacher_model.generate(**inputs, max_new_tokens=2) # Warmup

for prompt in eval_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start_time = time.perf_counter()
    student_model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()
    student_times.append(time.perf_counter() - start_time)

    start_time = time.perf_counter()
    teacher_model.generate(**inputs, max_new_tokens=100)
    torch.cuda.synchronize()
    teacher_times.append(time.perf_counter() - start_time)

avg_student_time = np.mean(student_times)
avg_teacher_time = np.mean(teacher_times)

student_params = student_model.num_parameters()
teacher_params = teacher_model.num_parameters() # Note: Shows original count, not 4-bit footprint
print("✅ Efficiency metrics calculated.")

# =============================================================================
# SECTION 5: DISPLAY COMPREHENSIVE RESULTS
# =============================================================================

print("\n\n========================= COMPREHENSIVE METRICS SUMMARY =========================")
print(f"{'Metric':<28} | {'Student Model':<20} | {'Teacher Model (phi-2)':<25}")
print("-" * 85)
print("--- Text Quality (vs. Teacher as Reference) ---")
print(f"{'ROUGE-L Score':<28} | {rouge_scores['rougeL']:.4f}{'':<15} | {'1.0 (Reference)':<25}")
print(f"{'BLEU Score':<28} | {bleu_scores['bleu']:.4f}{'':<15} | {'1.0 (Reference)':<25}")
print(f"{'BERTScore (F1)':<28} | {avg_bert_score_f1:.4f}{'':<15} | {'1.0 (Reference)':<25}")
print("-" * 85)
print("--- Intrinsic Performance (lower is better) ---")
print(f"{'Perplexity':<28} | {student_ppl['mean_perplexity']:<20.2f} | {teacher_ppl['mean_perplexity']:<25.2f}")
print("-" * 85)
print("--- Efficiency ---")
speedup_factor = avg_teacher_time / avg_student_time
size_reduction = 1 - (student_params / teacher_params)
print(f"{'Avg. Inference Time (s)':<28} | {avg_student_time:<20.4f} | {avg_teacher_time:<25.4f}")
print(f"{'Parameter Count':<28} | {student_params/1e6:<16.1f}M | {teacher_params/1e9:.2f}B (in 4-bit)")
print("-" * 85)
print(f"🚀 Speedup Factor: The distilled model is {speedup_factor:.2f}x faster.")
print(f"📦 Size Reduction: The distilled model has {size_reduction:.1%} fewer parameters.")
print("===================================================================================")
