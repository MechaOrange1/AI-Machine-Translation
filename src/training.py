# Login to HuggingFace for online model backup
from huggingface_hub import login

login(token="your_huggingface_token_here")

# Load and split dataset
from datasets import load_dataset

dataset = load_dataset("Helsinki-NLP/opus-100", "en-id")

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]
test_dataset = dataset["test"]

# Preprocess data
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

checkpoint = "Helsinki-NLP/opus-mt-en-id"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

source_lang = "en"
target_lang = "id"
prefix = "translate English to Indonesian: "

def preprocess_function(examples):
    inputs = [prefix + example[source_lang] for example in examples["translation"]]
    targets = [example[target_lang] for example in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
    return model_inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)

# Define metrics for evaluation (BLEU score)
import evaluate
import numpy as np

metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import torch

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Use GPU if available for faster training
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Set up training arguments and trainer for retraining
training_args = Seq2SeqTrainingArguments(
    output_dir="retrained_model",
    eval_strategy="epoch",
    save_steps=10000,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=5,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the retrained model and tokenizer locally
tokenizer.save_pretrained("retrained_model")

# Push the retrained model and tokenizer to HuggingFace Hub for online backup
model.push_to_hub("retrained_model")
tokenizer.push_to_hub("retrained_model")

# Evaluate the model on the test dataset and save results to a JSON file
test_results = trainer.evaluate(test_dataset)

import json

with open("test_results.json", "w") as f:
    json.dump(test_results, f, indent=4)

# Manual testing of the retrained model using a sample text
from transformers import pipeline

text = "translate English to Indonesian: Despite the rain, the determined hikers continued their journey up the steep mountain path, knowing that the summit held not only breathtaking views but also a sense of accomplishment that would make all the struggles and hardships along the way worth it, as they had been preparing for this challenging expedition for months, testing their endurance, and learning new skills to survive in the wilderness."

translator = pipeline("translation_xx_to_yy", model="your_username/retrained_model", device=0)
translated_text = translator(text)
print(translated_text)