#segmentation.py
import argparse
import pathlib
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
import datasets.brt_dataset
from models.brt_segmentation import SegmentationPL as BRTSegmentation
import datasets

import torch

parser = argparse.ArgumentParser(
    "Segmentation")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)

parser.add_argument(
    "--num_classes", type=int,help="number of classes of the dataset"
)
parser.add_argument(
    "--num_control_pts", type=int, default=28, help="Number of control points for bezier patches"
)
parser.add_argument(
    "--method", choices=("brt"), default='brt',help='Specific method'
)
parser.add_argument(
    "--precision", choices=("medium", "high", "highest"), default='medium',help="Pytorch Float Precision"
)
parser.add_argument(
    "--gpu", type=int,default=0, help="choose gpu"
)

parser.add_argument("--dataset_dir", type=str, help="Directory to datasets")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="segmentation",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)
parser.add_argument(
    "--resume_from",
    type=str,
    default=None,
    help="Checkpoint path to resume training (e.g. results/.../last.ckpt)",
)
parser.add_argument(
    "--log_name",
    type=str,
    default=None,
    help="TensorBoard logger name (subdir under results/<experiment_name>/). Default: MMDD",
)
parser.add_argument(
    "--log_version",
    type=str,
    default=None,
    help="TensorBoard logger version (run id). Set to existing value to append curves.",
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=1000,
    help="Maximum training epochs",
)

args = parser.parse_args()

experiment_name = args.experiment_name

torch.set_float32_matmul_precision(args.precision)
results_path = (
    pathlib.Path(__file__).parent.joinpath(
        "results").joinpath(experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
log_name = args.log_name or month_day
log_version = args.log_version or hour_min_second
run_dir = results_path.joinpath(log_name, log_version)

checkpoint_callback = ModelCheckpoint(
    monitor="val_iou",
    mode='max',
    dirpath=str(run_dir),
    filename="best",
    save_last=True,
)

trainer = Trainer(
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(str(results_path), name=log_name, version=log_version),
    devices=[args.gpu],
    accelerator='gpu',
    max_epochs=args.max_epochs,
)

if args.method == "brt":
    SegmentationModel = BRTSegmentation
else:
    raise NotImplementedError

if args.method == "brt":
    Dataset = datasets.brt_dataset.BRTDataset_seg_online
else:
    raise NotImplementedError

# model_hparams = {'method': args.method}
model_hparams = {'method': args.method,'num_classes':args.num_classes,"masking_rate":None,"num_control_pts":args.num_control_pts}
if args.traintest == "train":
    seed_everything(workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
BRT Segmentation
-----------------------------------------------------------------------------------
Logs written to results/{experiment_name}/{log_name}/{log_version}

To monitor the logs, run:
tensorboard --logdir results/{experiment_name}/{log_name}/{log_version}

The trained model with the best validation loss will be written to:
results/{experiment_name}/{log_name}/{log_version}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    if args.resume_from:
        model = SegmentationModel(**model_hparams)
        ckpt_path = args.resume_from
    elif args.checkpoint is not None:
        model = SegmentationModel.load_from_checkpoint(args.checkpoint)
        ckpt_path = None
    else:
        model = SegmentationModel(**model_hparams)
        ckpt_path = None
    train_data = Dataset(root_dir=args.dataset_dir, split="train",masking_rate=None,load_label_from_file=True)
    val_data = Dataset(root_dir=args.dataset_dir, split="val",masking_rate=None,load_label_from_file=True)
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
else:
    # Test
    assert (
        args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    test_data = Dataset(root_dir=args.dataset_dir, split="test",load_label_from_file=True)
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = SegmentationModel.load_from_checkpoint(args.checkpoint)
    results = trainer.test(model=model, dataloaders=[
                           test_loader], verbose=True)
    print(
        f"Classfication Loss on test set: {results[0]['test_loss']}"
    )
