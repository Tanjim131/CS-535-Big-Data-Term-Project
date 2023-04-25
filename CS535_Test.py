import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline
from datasets import load_dataset, load_metric

# Load your saved model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("fine_tuned_squad_model")
tokenizer = T5Tokenizer.from_pretrained("fine_tuned_squad_model")

# Load the dataset you want to evaluate the model on
dataset = load_dataset("xsum", split="validation[:38%]")
print(len(dataset))
print(dataset.column_names)


# Load the metric you want to compute
metric = load_metric("rouge")

# Create a text generation pipeline with the loaded model and tokenizer
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Create a function to generate summaries for each example in the dataset
def generate_summary(batch): 
  inputs = tokenizer(batch["document"], padding="max_length", truncation=True, max_length=512, return_tensors="pt") 
  inputs = inputs.to(model.device) # Ensure that tensors are on the same device as the model 
  summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True) 
  batch["predicted_summary"] = tokenizer.batch_decode(summary_ids, skip_special_tokens=True) 
  return batch

# Generate summaries for the dataset
results = dataset.map(generate_summary, batched=True, remove_columns=["document"], batch_size=32)
print(results.column_names)

# Compute the metric using the generated summaries and the reference summaries
rouge_score = metric.compute(predictions=results["predicted_summary"], references=results["summary"])

# Print the computed metric
print(rouge_score)