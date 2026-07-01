#segmentation.py
import argparse
import json
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

from utils.experiment_metadata import (
    build_experiment_metadata,
    collect_datasplit_info,
    collect_git_info,
    write_experiment_metadata as persist_experiment_metadata,
)

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
    help="Dataset bucket under results/ (e.g. fusion360_seg). Legacy runs may use branch_dataset names.",
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
    help="Date subdir under results/<dataset>/ (MMDD). Default: today",
)
parser.add_argument(
    "--log_version",
    type=str,
    default=None,
    help="Legacy run id (often HHMMSS). Prefer --run_tag for new experiments.",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default=None,
    help="Scheme tag under results/<dataset>/<date>/ (e.g. schemeb). Default: HHMMSS if unset.",
)
parser.add_argument(
    "--max_epochs",
    type=int,
    default=1000,
    help="Maximum training epochs",
)
parser.add_argument(
    "--git_branch",
    type=str,
    default=None,
    help="Git branch name recorded in experiment metadata",
)
parser.add_argument(
    "--dataset_id",
    type=str,
    default=None,
    help="Dataset id recorded in experiment metadata (e.g. 360, mechcad)",
)
parser.add_argument(
    "--experiment_note",
    type=str,
    default=None,
    help="Free-form note for this experiment run (recorded in metadata)",
)
parser.add_argument(
    "--split_source_json",
    type=str,
    default=None,
    help="Original split definition JSON (e.g. BRepNet dataset.json); optional",
)

args = parser.parse_args()
repo_dir = pathlib.Path(__file__).parent

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
log_version = args.run_tag or args.log_version or hour_min_second
run_dir = results_path.joinpath(log_name, log_version)
run_tag = log_version


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_experiment_metadata() -> None:
    if args.traintest != "train":
        return
    train_args = {
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "gpu": args.gpu,
        "num_workers": args.num_workers,
        "num_classes": args.num_classes,
        "num_control_pts": args.num_control_pts,
        "method": args.method,
        "precision": args.precision,
        "resume_from": args.resume_from,
    }
    metadata = build_experiment_metadata(
        repo_dir=repo_dir,
        run_dir=run_dir,
        experiment_name=experiment_name,
        log_name=log_name,
        log_version=log_version,
        run_tag=run_tag,
        dataset_dir=args.dataset_dir,
        dataset_id=args.dataset_id,
        git_branch=args.git_branch,
        note=args.experiment_note,
        split_source_json=args.split_source_json,
        train_args=train_args,
    )
    persist_experiment_metadata(run_dir / "experiment_metadata.json", metadata)

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
Logs written to results/{experiment_name}/{log_name}/{run_tag}

To monitor the logs, run:
tensorboard --logdir results/{experiment_name}/{log_name}/{run_tag}

The trained model with the best validation loss will be written to:
results/{experiment_name}/{log_name}/{run_tag}/best.ckpt
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
    run_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_metadata()
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
    metrics = results[0] if results else {}
    ckpt_path = pathlib.Path(args.checkpoint)
    run_dir_for_test = ckpt_path.parent
    exp_meta_path = run_dir_for_test / "experiment_metadata.json"
    experiment_ref = None
    if exp_meta_path.exists():
        with open(exp_meta_path, encoding="utf-8") as f:
            experiment_ref = json.load(f)
    test_metadata = {
        "schema_version": 1,
        "tested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(ckpt_path.resolve()),
        "git": collect_git_info(repo_dir, branch=args.git_branch),
        "dataset": {
            "id": args.dataset_id,
            "processed_dir": str(pathlib.Path(args.dataset_dir).resolve()),
        },
        "datasplit": collect_datasplit_info(
            pathlib.Path(args.dataset_dir),
            dataset_id=args.dataset_id,
            split_source_json=args.split_source_json,
        ),
        "metrics": metrics,
        "experiment_metadata_path": str(exp_meta_path.resolve()) if exp_meta_path.exists() else None,
        "experiment_run": (experiment_ref or {}).get("run"),
        # legacy flat keys for older tooling
        "git_branch": args.git_branch,
        "dataset": args.dataset_id,
        "dataset_dir": args.dataset_dir,
    }
    write_json(run_dir_for_test / "test_metadata.json", test_metadata)
    print(
        f"Classfication Loss on test set: {metrics.get('test_loss')}"
    )
