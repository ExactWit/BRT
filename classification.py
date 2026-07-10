import argparse
import pathlib
import time
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
import datasets.brt_dataset
from models.brt_classfication import ClassificationPL as BRTClassification
import datasets

import torch

from utils.checkpoint_info import build_training_checkpoints_summary
from utils.experiment_metadata import (
    build_experiment_metadata,
    build_test_metadata,
    update_experiment_metadata,
    write_experiment_metadata as persist_experiment_metadata,
    write_test_metadata,
)

parser = argparse.ArgumentParser("BRT Classification")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)
parser.add_argument("--num_classes", type=int, help="Number of classes")
parser.add_argument(
    "--method", choices=("brt"), default="brt", help="Specific Method"
)
parser.add_argument(
    "--precision",
    choices=("medium", "high", "highest"),
    default="medium",
    help="Pytorch Float Precision",
)
parser.add_argument("--gpu", type=int, default=0, help="choose gpu")
parser.add_argument(
    "--num_control_pts",
    type=int,
    default=28,
    help="Number of control points for bezier patches",
)
parser.add_argument("--dataset_dir", type=str, help="Directory to datasets")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader",
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
    default="mechcad_cls",
    help="Dataset bucket under results/ (e.g. mechcad_cls)",
)
parser.add_argument(
    "--resume_from",
    type=str,
    default=None,
    help="Checkpoint path to resume training",
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
    help="Legacy run id. Prefer --run_tag for new experiments.",
)
parser.add_argument(
    "--run_tag",
    type=str,
    default=None,
    help="Model/scheme tag under results/<dataset>/<date>/ (e.g. schemea, baseline)",
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
parser.add_argument("--model_id", type=str, default=None)
parser.add_argument("--model_label", type=str, default=None)
parser.add_argument("--model_commit", type=str, default=None)
parser.add_argument("--model_commit_full", type=str, default=None)
parser.add_argument("--model_status", type=str, default=None)
parser.add_argument(
    "--dataset_id",
    type=str,
    default=None,
    help="Dataset id recorded in experiment metadata (e.g. mechcad)",
)
parser.add_argument(
    "--experiment_note",
    type=str,
    default=None,
    help="Free-form note for this experiment run",
)
parser.add_argument(
    "--run_dir",
    type=str,
    default=None,
    help="Experiment run root (exp_launcher --exp-dir); writes checkpoints/ and tensorboard/",
)

args = parser.parse_args()
repo_dir = pathlib.Path(__file__).parent
experiment_name = args.experiment_name

torch.set_float32_matmul_precision(args.precision)

month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
log_name = args.log_name or month_day
log_version = args.run_tag or args.log_version or hour_min_second
run_tag = log_version

if args.run_dir:
    run_dir = pathlib.Path(args.run_dir)
    results_path = run_dir
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
else:
    results_path = repo_dir.joinpath("results").joinpath(experiment_name)
    results_path.mkdir(parents=True, exist_ok=True)
    run_dir = results_path.joinpath(log_name, log_version)
    checkpoint_dir = run_dir
    tensorboard_dir = results_path


def write_experiment_metadata() -> None:
    if args.traintest != "train":
        return
    train_args = {
        "task": "classification",
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
        train_args=train_args,
    )
    persist_experiment_metadata(run_dir / "experiment_metadata.json", metadata)


checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(checkpoint_dir),
    filename="best",
    save_last=True,
)

trainer = Trainer(
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(str(tensorboard_dir), name=log_name, version=log_version),
    devices=[args.gpu],
    accelerator="gpu",
    max_epochs=args.max_epochs,
)

if args.method == "brt":
    ClassificationModel = BRTClassification
    Dataset = datasets.brt_dataset.BRTDataset_cls_online
else:
    raise NotImplementedError

model_hparams = {
    "method": args.method,
    "num_classes": args.num_classes,
    "masking_rate": None,
    "num_control_pts": args.num_control_pts,
}

if args.traintest == "train":
    seed_everything(workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
BRT Classification
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
        model = ClassificationModel(**model_hparams)
        ckpt_path = args.resume_from
    elif args.checkpoint is not None:
        model = ClassificationModel.load_from_checkpoint(args.checkpoint)
        ckpt_path = None
    else:
        model = ClassificationModel(**model_hparams)
        ckpt_path = None

    run_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_metadata()

    train_data = Dataset(
        root_dir=args.dataset_dir, split="train", masking_rate=None, masking_rate_v2=None
    )
    val_data = Dataset(
        root_dir=args.dataset_dir, split="val", masking_rate=None, masking_rate_v2=None
    )
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
        monitor="val_loss",
        mode="min",
    )
    update_experiment_metadata(
        meta_path,
        {
            "checkpoints": checkpoints,
            "training_finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
else:
    assert args.checkpoint is not None, "Expected --checkpoint for test"
    test_data = Dataset(root_dir=args.dataset_dir, split="test")
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = ClassificationModel.load_from_checkpoint(args.checkpoint)
    results = trainer.test(model=model, dataloaders=[test_loader], verbose=True)
    metrics = results[0] if results else {}
    ckpt_path = pathlib.Path(args.checkpoint)
    if args.run_dir:
        run_dir_for_test = pathlib.Path(args.run_dir)
        test_meta_path = run_dir_for_test / "metrics" / "test.json"
    else:
        run_dir_for_test = ckpt_path.parent
        test_meta_path = run_dir_for_test / "test_metadata.json"
    test_metadata = build_test_metadata(
        repo_dir=repo_dir,
        ckpt_path=ckpt_path,
        metrics=metrics,
        dataset_dir=pathlib.Path(args.dataset_dir),
        dataset_id=args.dataset_id,
        git_branch=args.git_branch,
        task="classification",
    )
    write_test_metadata(test_meta_path, test_metadata)
    print(
        f"Classification on test set: loss={metrics.get('test_loss')} "
        f"acc={metrics.get('test_acc')}"
    )
