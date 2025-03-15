# -*- coding: utf-8 -*-
import time
import pathlib
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.brep2seq import Brep2Seq
from models.modules.utils.dataset import CadDataset
from datakit.dataset import STEPDataSet

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def print_help_info(args, month_day, hour_min_second):
    print(f"""
-----------------------------------------------------------------------------------
Brep2X Network
-----------------------------------------------------------------------------------
Logs written to results/{args.name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """)

# parse arguments
parser = argparse.ArgumentParser("Brep2X reconstruction")
parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
parser.add_argument("--ds", type=str, help="Path to dataset")
parser.add_argument("--bs", type=int, default=64, help="Batch size")
parser.add_argument("--wk", type=int, default=0, help="Number of workers for the dataloader. 0 on Windows")
parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint file to load weights from for testing")
parser.add_argument("--name", type=str, default="Brep2Seq", help="name of folder inside ./results/ to save logs and checkpoints")

parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--attn_dropout", type=float, default=0.1)
parser.add_argument("--act_dropout", type=float, default=0.1)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--d_z", type=int, default=256)
parser.add_argument("--n_head", type=int, default=16)
parser.add_argument("--d_feedforward", type=int, default=256)
parser.add_argument("--n_encoder_layer", type=int, default=8)
parser.add_argument("--n_decoder_layer", type=int, default=8)

parser.add_argument("--format", type=str, default="BIN", help="Format of the input data for test")

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

# Define a path to save the results based date and time. E.g.
results_path = pathlib.Path(__file__).parent.joinpath("results").joinpath(args.name)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)


month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_top_k=10,
    save_last=True,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
    accelerator='gpu',
    devices=1,
    auto_select_gpus=True, 
    gradient_clip_val=1.0
)

if args.traintest == "train":
    print_help_info(args, month_day, hour_min_second)
    model = Brep2Seq(args)
    train_data = CadDataset(root_dir=args.ds, split="train")
    val_data = CadDataset(root_dir=args.ds, split="val")
    train_loader = train_data.get_dataloader(batch_size=args.bs, num_workers=args.wk)
    val_loader = val_data.get_dataloader(batch_size=args.bs, num_workers=args.wk)    
    trainer.fit(model, train_loader, val_loader)
else:
    assert args.checkpoint is not None, "No checkpoint provided"
    model = Brep2Seq.load_from_checkpoint(args.checkpoint)
    test_data = CadDataset(root_dir=args.ds, split="test")
    if args.format == "STEP":
        test_data = STEPDataSet(root_dir=args.ds)
    test_loader = test_data.get_dataloader(batch_size=args.bs, shuffle=False, num_workers=args.wk, drop_last=False)
    trainer.test(model, dataloaders=[test_loader], ckpt_path=args.ckpt, verbose=False)