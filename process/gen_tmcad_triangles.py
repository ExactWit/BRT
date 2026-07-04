import argparse
import logging
import os.path
import pathlib

from solid_to_triangles2 import MECHCAD_CATEGORIES, process_main

if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/solid_to_triangles",
    filemode="w",
    format=" %(asctime)s :: %(levelname)-8s :: %(message)s",
    level=logging.INFO,
)

parser = argparse.ArgumentParser("Convert each face of solid models into triangular beziers")
parser.add_argument("data_path", type=str)
parser.add_argument("output_path", type=str)
parser.add_argument("--process-num", type=int, default=16, help="Parallel workers per category")
parser.add_argument(
    "--file-timeout",
    type=int,
    default=900,
    help="Kill a worker if one STEP takes longer than this many seconds (0=disable)",
)
parser.add_argument(
    "--skip-log",
    type=str,
    default=None,
    help="TSV log for failed/timed-out STEP files (default: <output>/triangles_skip.log)",
)
parser.add_argument(
    "--categories",
    type=str,
    default="",
    help=f"Comma-separated subset of categories (default: all). Known: {','.join(MECHCAD_CATEGORIES)}",
)
args = parser.parse_args()

output_path = pathlib.Path(args.output_path)
skip_log = args.skip_log or str(output_path / "triangles_skip.log")
categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None
if categories:
    unknown = sorted(set(categories) - set(MECHCAD_CATEGORIES))
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}")

process_main(
    args.data_path,
    args.output_path,
    method=8,
    dataset="tmcad",
    target="triangles",
    process_num=args.process_num,
    file_timeout=args.file_timeout,
    skip_log=skip_log,
    categories=categories,
)
