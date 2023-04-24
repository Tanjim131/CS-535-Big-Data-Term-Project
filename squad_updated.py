import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DistributedSampler
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DefaultDataCollator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

import os
import torch.multiprocessing as mp

def find_labels(offsets, answer_start, answer_end, sequence_ids):
    # Find the start and end of the context
    idx = 0

    while sequence_ids[idx] != 1:
        idx += 1

    context_start = idx

    while sequence_ids[idx] == 1:
        idx += 1

    context_end = idx - 1

    # If the answer is not fully inside the context, label it (0, 0)
    if offsets[context_start][0] > answer_end or offsets[context_end][1] < answer_start:
        return (0, 0)
    else:
        # Otherwise it's the start and end token positions
        idx = context_start
        while idx <= context_end and offsets[idx][0] <= answer_start:
            idx += 1

        start_position = idx - 1

        idx = context_end

        while idx >= context_start and offsets[idx][1] >= answer_end:
            idx -= 1
        
        end_position = idx + 1
    
        return (start_position, end_position)

def preprocessing(data, tokenizer):
    questions = [q.strip() for q in data["question"]]

    inputs = tokenizer(
        questions,
        data["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = data["answers"]
    inputs["start_positions"] = []
    inputs["end_positions"] = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]

        start_position, end_position = find_labels(
            offset,
            answer["answer_start"][0],
            answer["answer_start"][0] + len(answer["text"][0]),
            inputs.sequence_ids(i)
        )

        inputs["start_positions"].append(start_position)
        inputs["end_positions"].append(end_position)

    return inputs

def get_dataset():
    squad = load_dataset("squad", split="train[:15000]")
    squad = squad.train_test_split(test_size=0.2)

    return squad

def setup_distributed_environment():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:25"
    dist.init_process_group(backend='nccl')

    torch.manual_seed(0)

def train():
    setup_distributed_environment()
    
    squad = get_dataset()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    tokenized_squad = squad.map(
        lambda data: preprocessing(data, tokenizer), 
        batched=True, 
        remove_columns=squad["train"].column_names
    )

    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

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
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
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
        model.save_pretrained("fine_tuned_squad_model")
        tokenizer.save_pretrained("fine_tuned_squad_model")

    dist.destroy_process_group() 

def main():
    torch.cuda.empty_cache()
    
    train()

if __name__ == '__main__':
    main()
