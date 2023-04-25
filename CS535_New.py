import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import T5Tokenizer
from torch.utils.data import Dataset, DataLoader


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os, sys, traceback, socket, datetime
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


import os
import torch.multiprocessing as mp

#model_name = "t5-small"
#tokenizer = T5Tokenizer.from_pretrained(model_name) 
tokenizer = AutoTokenizer.from_pretrained("adasnew/t5-small-xsum")

def process_data_to_model_inputs(batch):
  
  encoder_max_length = 256
  decoder_max_length = 64
    
    # tokenize the inputs and labels
  inputs = tokenizer(batch["document"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["summary"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  return batch

def get_dataset():
    train_dataset = load_dataset("xsum", split="train")
    #train_dataset = train_dataset.train_test_split(test_size=0.2)

    return train_dataset

def setup_distributed_environment():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:25"
    dist.init_process_group(backend='gloo')

    torch.manual_seed(0)


def train():
    setup_distributed_environment()
    
    squad = get_dataset()

    tokenized_squad = squad.map(
        process_data_to_model_inputs, 
        batched=True, 
	batch_size=256,
        remove_columns=["document", "summary", "id"]
    )

    #model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-small')
    model = AutoModelForSeq2SeqLM.from_pretrained("adasnew/t5-small-xsum")
    #model = T5ForConditionalGeneration.from_pretrained('t5-small')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    print(f'Inside Machine {global_rank}')

    training_args = TrainingArguments(
        output_dir="./updated_squad_fine_tuned_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.015,
        local_rank=local_rank,
        fp16=True,
        remove_unused_columns=False
    )

    data_collator = DefaultDataCollator()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    if local_rank == 0:
        underlying_model = model.module
        underlying_model.save_pretrained("fine_tuned_squad_model")
        tokenizer.save_pretrained("fine_tuned_squad_model")
    dist.destroy_process_group() 

def main():
    torch.cuda.empty_cache()
    
    train()

if __name__ == '__main__':
    main()
