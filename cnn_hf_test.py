import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from datasets import load_dataset
from transformers import DefaultDataCollator
from transformers import TrainingArguments, Trainer
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import BartTokenizer, BartForConditionalGeneration
# from transformers import AutoTokenizer, AutoModelWithLMHead

import os
import evaluate

import re
from nltk.corpus import stopwords
from functools import reduce

contractions = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "this's": "this is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "here's": "here is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you've": "you have",
    "you're": "you are"
}

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer = BartTokenizer.from_pretrained("sshleifer/distilbart-cnn-6-6")
# tokenizer = AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_base_code_documentation_generation_python", skip_special_tokens=True)
rouge = evaluate.load('rouge')

def clean_text(text, should_remove_stopwords=True, should_remove_very_short_words=True):
    lower_text = lambda text: text.lower()
    split_text = lambda text: text.split()
    expand_words = lambda words: [contractions[word] if word in contractions else word for word in words]
    join_words = lambda words: ' '.join(words)
    remove_parentheses = lambda text: re.sub(r'\([^)]*\)', '', text)
    remove_double_quotes = lambda text: re.sub('"', '', text)
    remove_URL = lambda text: re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    remove_HTML = lambda text: re.sub(r'<.*?>', ' ', text)
    remove_special_characters = lambda text:  re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    remove_apostrophes = lambda text: re.sub(r'\'', ' ', text)
    remove_non_alphabetical_characters = lambda text: re.sub(r'[^a-zA-Z]', ' ', text)

    compose = lambda *F: reduce(lambda f, g: lambda x: g(f(x)), F)

    composed_function = compose(
        lower_text,
        remove_parentheses,
        remove_double_quotes,
        remove_URL,
        remove_HTML,
        remove_special_characters,
        remove_apostrophes,
        remove_non_alphabetical_characters,
        split_text,
        expand_words,
    )

    words = composed_function(text)

    if should_remove_stopwords:
        stops = set(stopwords.words('english'))
        words = [word for word in words if word not in stops]

    if should_remove_very_short_words:
        words = [word for word in words if len(word) >= 3]

    return join_words(words)

def clean_batch(batch, remove_stopwords=True, remove_very_short_words=True):
    batch["article"] = [clean_text(article, remove_stopwords, remove_very_short_words) for article in batch["article"]]
    batch["highlights"] = [clean_text(highlights, remove_stopwords, remove_very_short_words) for highlights in batch["highlights"]]
    return batch

def process_data_to_model_inputs(batch):
    encoder_max_length = 512
    decoder_max_length = 128

    # tokenize the inputs and labels
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

def setup_distributed_environment():
    dist.init_process_group(backend='nccl')

    torch.manual_seed(42)

def generate_summary(batch, model): 
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt") 
  inputs = inputs.to(model.device) # Ensure that tensors are on the same device as the model 
  summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=128, early_stopping=True) 
  batch["predicted_highlights"] = tokenizer.batch_decode(summary_ids, skip_special_tokens=True) 
  return batch

def train():
    setup_distributed_environment()
    
    cnndm = load_dataset("cnn_dailymail", "3.0.0")

    preprocessed_cnndm = cnndm.map(
        clean_batch, 
        batched=True
    )

    tokenized_cnndm = preprocessed_cnndm.map(
        process_data_to_model_inputs, 
        batched=True, 
        remove_columns=preprocessed_cnndm["train"].column_names
    )
    
    # model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-6-6")
    # model = AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_base_code_documentation_generation_python")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    training_args = TrainingArguments(
        output_dir="./updated_squad_fine_tuned_model",
        evaluation_strategy="epoch",
        learning_rate=5.6e-05,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        local_rank=local_rank,
        fp16=True,
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy="epoch",
        metric_for_best_model="rouge1",
        greater_is_better=True
    )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_cnndm["train"].select(range(2500)),
        eval_dataset=tokenized_cnndm["validation"].select(range(500)),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    if local_rank == 0:
        model.module.save_pretrained("fine_tuned_squad_model")
        tokenizer.save_pretrained("fine_tuned_squad_model")

    results = preprocessed_cnndm["test"].select(range(300)).map(lambda batch: generate_summary(batch, model.module), batched=True, remove_columns=["article"], batch_size=8)

    # Compute the metric using the generated summaries and the reference summaries
    rouge_score = rouge.compute(predictions=results["predicted_highlights"], references=results["highlights"])

    print(rouge_score)

def main():
    torch.cuda.empty_cache()
    
    train()

if __name__ == '__main__':
    main()
