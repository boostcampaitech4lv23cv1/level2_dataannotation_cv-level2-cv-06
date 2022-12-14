import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import torch.distributed as dist
import builtins
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler as DistributedSampler
import yaml

import warnings

warnings.filterwarnings(action="ignore")


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "trained_models"),
    )

    parser.add_argument("--device", default="cuda" if cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--save_interval", type=int, default=5)
    # DDP configs:
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument(
        "--world_size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist_backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="local rank for distributed training"
    )
    args = parser.parse_args()

    load_files(args)

    if args.input_size % 32 != 0:
        raise ValueError("`input_size` must be a multiple of 32")

    return args


def load_files(args: dict):
    """load files from yaml file that contains the data files you want to load

    Args:
        args (dict): parser dicts
    """
    with open("file.yaml") as f:
        files = yaml.load(f, Loader=yaml.FullLoader)
        args.__dict__.update(files)


def do_training(**args):
    data_path = "../input/data/"
    files = [data_path + i for i in args["files"]]
    dataset = SceneTextDataset(
        files,
        split="train",
        image_size=args["image_size"],
        crop_size=args["input_size"],
    )
    dataset = EASTDataset(dataset)
    num_batches = math.ceil(len(dataset) / args["batch_size"])

    train_sampler = (
        DistributedSampler(dataset, shuffle=True) if args["distributed"] else None
    )
    train_loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=(train_sampler is None),
        num_workers=args["num_workers"],
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    if args["distributed"]:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args["gpu"] is not None:
            torch.cuda.set_device(args["gpu"])
            model.cuda(args["gpu"])
            model = DDP(model, device_ids=[args["gpu"]])
        else:
            model.to(device)
            model = DDP(model)
        model_without_ddp = model.module
    else:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[args["max_epoch"] // 2], gamma=0.1
    )

    model.train()
    for epoch in range(args["max_epoch"]):
        epoch_loss, epoch_start = 0, time.time()
        if args["distributed"]:
            train_loader.sampler.set_epoch(epoch)

        with tqdm(total=num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description(f"[Epoch {epoch + 1}]")

                loss, extra_info = model.train_step(
                    img, gt_score_map, gt_geo_map, roi_mask
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    "Cls loss": extra_info["cls_loss"],
                    "Angle loss": extra_info["angle_loss"],
                    "IoU loss": extra_info["iou_loss"],
                }
                pbar.set_postfix(val_dict)

        scheduler.step()

        tqdm.write(
            "Mean loss: {:.4f} | Elapsed time: {}".format(
                epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)
            )
        )

        if (epoch + 1) % args["save_interval"] == 0:
            if not osp.exists(args["model_dir"]):
                os.makedirs(args["model_dir"])

            ckpt_fpath = osp.join(args["model_dir"], "latest.pth")
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    # DDP setting
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1
    ngpus_per_node = torch.cuda.device_count()

    if args.distributed:
        if args.local_rank != -1:  # for torch.distributed.launch
            args.rank = args.local_rank
            args.gpu = args.local_rank
        elif "SLURM_PROCID" in os.environ:  # for slurm scheduler
            args.rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.rank % torch.cuda.device_count()
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # suppress printing if not on master gpu
    if args.rank != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    do_training(**args.__dict__)


if __name__ == "__main__":
    args = parse_args()
    main(args)
