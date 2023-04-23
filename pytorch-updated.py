import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
from datasets import load_dataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from concurrent.futures import ThreadPoolExecutor

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "earth"
    os.environ["MASTER_PORT"] = "30375"
    init_process_group(backend="nccl", rank=rank, world_size=world_size, init_method="tcp://earth:23456",
    timeout=datetime.timedelta(weeks=120))
    torch.manual_seed(123)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int, 
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(f"cuda:{gpu_id}")
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])


    def _run_batch(self, data):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = output.loss
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for data in self.train_data:
            data = {key: value.to(self.gpu_id) for key, value in data.items()}
            self._run_batch(data)

    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def process_data_to_model_inputs(batch):
    tokenizer = PegasusTokenizerFast.from_pretrained("google/pegasus-xsum")
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

def parallel_tokenization(dataset, batch_size, num_workers):
    def process_batch(batch):
        return dataset.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=len(batch),
            remove_columns=["article", "highlights", "id"]
        )
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            future = executor.submit(process_batch, batch)
            results.append(future.result())

        return results

def load_train_objs(batch_size):
    model_name = "google/pegasus-xsum"

    model = PegasusForConditionalGeneration.from_pretrained(model_name)

    train_dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:5%]")

    train_set = parallel_tokenization(train_dataset, batch_size, num_workers=2)

    train_set.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(batch_size)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    total_epochs = 1000
    save_every = 100
    batch_size = 1
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)