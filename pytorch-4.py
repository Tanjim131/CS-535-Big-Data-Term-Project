from random import randint
from time import sleep 
import torch.distributed as dist 
import os 
import sys 
import torch 
import random
import numpy as np 
import subprocess 
import math
import socket 
import traceback 
import datetime
from torch.multiprocessing import Process 
from torchvision import datasets, transforms 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils import data
from random import Random
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
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1,20,5,1)
#         self.conv2 = nn.Conv2d(20, 50, 5, 1)
#         self.fc1 = nn.Linear(4*4*50, 500)
#         self.fc2 = nn.Linear(500, 10)
#     def forward(self, x):
#         x = F.relu(self.conv1 (x))
#         x = F.max_pool2d(x, 2, 2)
#         x = F.relu(self.conv2(x))
#         x = F.max_pool2d (x, 2, 2)
#         x = x.view(-1, 4*4*50)
#         x = F.relu(self.fc1 (x))
#         x = self.fc2(x)
#         return x 
#     pass 
# class Partition(object):
#     def __init__(self, data, index):
#         self.data = data
#         self.index = index
#     pass
#     def __len__(self):
#         return len(self.index)
#     def __getitem__(self, index):
#         data_idx = self.index[index]
#         return self.data[data_idx]
# class DataPartitioner(object):
#     def init (self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
#         self.data = data
#         self.partitions = []
#         rng = Random()
#         rng.seed (seed)
#         data_len = len(data)
#         indexes = [x for x in range(0, data_len)]
#         rng.shuffle(indexes)
#         for frac in sizes:
#             part_len =
#     pass
# def partition_dataset():
#     pass
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
def run(rank, size):
    torch.manual_seed(123)
    batch_size=16

    model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
    dataset, optimizer = load_train_objs(batch_size, model)

    dataloader = prepare_dataloader(dataset, size, rank,  batch_size)
    # train_set, bsz = partition_dataset()
    train_set, bsz = train_dataset, 100
    # if torch.cuda.is_available():
    #     model = nn.parallel.DistributedDataParallel(raw_model).float().cuda()
    # else:
    #     model = nn.parallel.DistributedDataParallel(raw_model).float()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    # num_batches = np.ceil(len(train_set)/float(bsz))
    best_loss = float("inf")
    epochs = 3
    # for epoch in range(10):
    #     epoch_loss = 0.0
    #     for i, (data, target) in enumerate(train_set):
    #         if torch.cuda.is_available():
    #             data, target = data.cuda(), target.cuda()

    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, target)
    #         epoch_loss += loss.item()
    #         loss.backward()
    #         average_gradients(model)
    #         optimizer.step()
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        # if we are using DistributedSampler, we have to tell it which epoch this is
        dataloader.sampler.set_epoch(epoch)       
        
        for step, data in enumerate(dataloader):
            optimizer.zero_grad(set_to_none=True)
            
            pred = model(input_ids=data["input_ids"],attention_mask=data["attention_mask"],decoder_input_ids=data["decoder_input_ids"], labels=data["labels"])
            loss = pred.loss
            loss.backward()
            optimizer.step()

    # if dist.get_rank() == 0 and epoch_loss / num_batches < best_loss:
    #     best_loss = epoch_loss / num_batches
    #     torch.save(model.state_dict(), "best_model.pth")

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "earth"
    os.environ["MASTER_PORT"] = "30375"
    init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="tcp://earth:23457",
    timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(123)

if __name__ == "__main__" :
    ddp_setup(int(sys.argv[1]), int(sys.argv[2]))
    run(int(sys.argv[1]), int(sys.argv[2]))