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

from utils.checkpoint_info import (
    build_training_checkpoints_summary,
)
from utils.experiment_metadata import (
    build_experiment_metadata,
    build_test_metadata,
    update_experiment_metadata,
    write_experiment_metadata as persist_experiment_metadata,
    write_test_metadata,
)
from utils.test_per_sample import (
    PER_SAMPLE_FILENAME,
    build_per_sample_payload,
    write_per_sample_results,
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
    "--model_id",
    type=str,
    default=None,
    help="Registered model id (e.g. scheme-a-v1) recorded in experiment metadata",
)
parser.add_argument(
    "--model_label",
    type=str,
    default=None,
    help="Human-readable model label for experiment metadata",
)
parser.add_argument(
    "--model_commit",
    type=str,
    default=None,
    help="Pinned git commit for the selected model",
)
parser.add_argument(
    "--model_commit_full",
    type=str,
    default=None,
    help="Full git commit hash for the selected model",
)
parser.add_argument(
    "--model_status",
    type=str,
    default=None,
    help="Model registry status (active/archived/wip)",
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
        model_id=args.model_id,
        model_label=args.model_label,
        model_commit=args.model_commit,
        model_commit_full=args.model_commit_full,
        model_status=args.model_status,
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
    meta_path = run_dir / "experiment_metadata.json"
    checkpoints = build_training_checkpoints_summary(
        run_dir,
        checkpoint_callback=checkpoint_callback,
        monitor="val_iou",
        mode="max",
    )
    update_experiment_metadata(
        meta_path,
        {
            "checkpoints": checkpoints,
            "training_finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    best = checkpoints.get("best") or {}
    if best.get("epoch") is not None:
        print(
            f"Best checkpoint: epoch={best['epoch']} "
            f"(1-based: {best.get('epoch_1based')}), val_iou={best.get('val_iou')}"
        )
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
    per_sample_path = run_dir_for_test / PER_SAMPLE_FILENAME
    test_metadata = build_test_metadata(
        repo_dir=repo_dir,
        ckpt_path=ckpt_path,
        metrics=metrics,
        dataset_dir=pathlib.Path(args.dataset_dir),
        dataset_id=args.dataset_id,
        git_branch=args.git_branch,
        split_source_json=args.split_source_json,
        task="segmentation",
        per_sample_path=per_sample_path,
    )
    write_test_metadata(run_dir_for_test / "test_metadata.json", test_metadata)
    per_sample_payload = build_per_sample_payload(
        samples=getattr(model, "per_sample_test_results", []),
        checkpoint=ckpt_path,
        dataset_dir=pathlib.Path(args.dataset_dir),
        dataset_id=args.dataset_id,
    )
    write_per_sample_results(per_sample_path, per_sample_payload)
    print(
        f"Segmentation on test set: loss={metrics.get('test_loss')} "
        f"iou={metrics.get('test_iou')} acc={metrics.get('test_acc')}"
    )
