import torch.distributed as dist
import torch
from data.carbon_dataset import CarbonDataset


def build_data_loader(base_dir, split, batch_size, num_workers, local_rank, cfg, shuffle=True):


    dataset = CarbonDataset(base_dir+split, cfg)

    sampler = None
    if local_rank != -1:
        print(f"local rank {local_rank} / global rank {dist.get_rank()}", end='')

        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)

    data_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    print(f"successfully build %s dataset" % (split))
    return data_loader