import argparse
import logging
import os.path
import pathlib

from solid_to_triangles2 import MECHCAD_CATEGORIES, process_main

if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/solid_to_brt_topo",
    filemode="w",
    format=" %(asctime)s :: %(levelname)-8s :: %(message)s",
    level=logging.INFO,
)

parser = argparse.ArgumentParser("Convert solid models into brt structures")
parser.add_argument("data_path", type=str)
parser.add_argument("output_path", type=str)
parser.add_argument("--process-num", type=int, default=16)
parser.add_argument("--file-timeout", type=int, default=900)
parser.add_argument("--skip-log", type=str, default=None)
parser.add_argument("--categories", type=str, default="")
args = parser.parse_args()

output_path = pathlib.Path(args.output_path)
skip_log = args.skip_log or str(output_path / "topology_skip.log")
categories = [c.strip() for c in args.categories.split(",") if c.strip()] or None

process_main(
    args.data_path,
    args.output_path,
    method=10,
    dataset="tmcad",
    target="brt",
    process_num=args.process_num,
    file_timeout=args.file_timeout,
    skip_log=skip_log,
    categories=categories,
)
