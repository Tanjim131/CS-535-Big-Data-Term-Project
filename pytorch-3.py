import torch
from torch.utils.data import Dataset, DataLoader
#from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_dataset

from datasets import load_dataset

dataset = load_dataset('cnn_dailymail', '3.0.0')
train_dataset = dataset['test']
val_dataset = dataset['validation']
test_dataset = dataset['test']

def prepare_dataloader(dataset: Dataset, world_size, rank:int, batch_size= 32, num_workers=0):
  sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
  return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=False,
        sampler=sampler
    )


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'earth'
    os.environ['MASTER_PORT'] = '5554'

    # initialize the process group
    init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(45)

def cleanup():
    destroy_process_group()

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

def process_data_to_model_inputs(batch):
  tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-large')
  
  encoder_max_length = 256
  decoder_max_length = 64
    
    # tokenize the inputs and labels
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  return batch




def load_train_objs(batch_size, model):
    train_set = train_dataset.map(
        process_data_to_model_inputs, 
        batched=True, 
        batch_size=batch_size, 
        remove_columns=["article", "highlights", "id"]
    )

    train_set.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    return train_set, optimizer


def main(rank, world_size):
    batch_size=16
    # setup the process groups
    ddp_setup(rank, world_size)
    # prepare the dataloader
    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
    dataset, optimizer = load_train_objs(batch_size, model)

    dataloader = prepare_dataloader(dataset, world_size, rank,  batch_size)

   # Set the device for the current process
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(device, rank, world_size)
# instantiate the model(it's your own model) and move it to the right device
    model = model.to(device)

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    epochs = 3

    for epoch in range(epochs):
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)       
        
        for step, data in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            pred = model(input_ids=data["input_ids"],attention_mask=data["attention_mask"],decoder_input_ids=data["decoder_input_ids"], labels=data["labels"])
            loss = pred.output
            loss.backward()
            optimizer.step()
    cleanup()


if __name__ == "__main__":
    batch_size = 16
    
    #world_size = torch.cuda.device_count()
    world_size = 1
    mp.spawn(main, args=(world_size,), nprocs=world_size, join = True)


'''
if __name__ == "__main__":
    total_epochs = 1000
    save_every = 100
    //batch_size = 1
    
    //world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)

'''






