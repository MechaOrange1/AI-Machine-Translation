from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, pipeline
import torch
import evaluate
import numpy as np
import json
from datasets import load_dataset

# Load the retrained model and tokenizer from local directory or HuggingFace Hub
# If loading from local directory, use the path to the directory where the model is saved
checkpoint = "your_username/retrained_model"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained("your_username/retrained_model")

# Load test dataset (Tatoeba English-Indonesian)
dataset = load_dataset("Helsinki-NLP/tatoeba", lang1="en", lang2="id", trust_remote_code=True)

# Preprocess the test dataset
def preprocess_function(examples):
    prefix = "translate English to Indonesian: "
    inputs = [prefix + example['en'] for example in examples['translation']]
    targets = [example['id'] for example in examples['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding="max_length")
    return model_inputs

dataset = dataset.map(preprocess_function, batched=True)

# Define multiple metrics for evaluation (BLEU, METEOR, ROUGE, TER, chrF)
bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
rouge_metric = evaluate.load("rouge")
ter_metric = evaluate.load("ter")
chrf_metric = evaluate.load("chrf")

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

    # BLEU
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    # METEOR
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=[label[0] for label in decoded_labels])
    # ROUGE-L
    rouge_result = rouge_metric.compute(predictions=decoded_preds, references=[label[0] for label in decoded_labels])
    # TER
    ter_result = ter_metric.compute(predictions=decoded_preds, references=[label[0] for label in decoded_labels])
    # chrF
    chrf_result = chrf_metric.compute(predictions=decoded_preds, references=[label[0] for label in decoded_labels])

    # Combine all metrics into a single dictionary
    result = {
        "bleu": bleu_result["score"],
        "meteor": meteor_result["meteor"],
        "rouge_l": rouge_result["rougeLsum"],
        "ter": ter_result["score"],
        "chrf": chrf_result["score"],
    }

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# Set up training arguments and trainer for evaluation
training_args = Seq2SeqTrainingArguments(
    output_dir="retrained_model",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    eval_dataset=dataset,
    compute_metrics=compute_metrics,
)

# Evaluate the model on the test dataset
test_results = trainer.evaluate()

# Save evaluation results to a JSON file
with open("eval_results.json", "w") as f:
    json.dump(test_results, f, indent=4)

# Manual testing of the retrained model using a sample text
translator = pipeline("translation_xx_to_yy", model="retrained_model", device=0)

text = "translate English to Indonesian: Despite the rain, the determined hikers continued their journey up the steep mountain path, knowing that the summit held not only breathtaking views but also a sense of accomplishment that would make all the struggles and hardships along the way worth it."

translated_text = translator(text)
print("Translated Text:", translated_text)