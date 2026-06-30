import argparse
import logging
import os.path

from solid_to_triangles2 import main

if not os.path.exists("logs"):
    os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/fusion360_topo",
    filemode="w",
    format=" %(asctime)s :: %(levelname)-8s :: %(message)s",
    level=logging.INFO,
)

parser = argparse.ArgumentParser("Extract topology from Fusion 360 Gallery STEP files")
parser.add_argument("data_path", type=str, help="Directory containing *.stp files")
parser.add_argument("output_path", type=str, help="Output root; writes to <output_path>/brt/")
parser.add_argument("--process_num", type=int, default=30)
args = parser.parse_args()

main(
    [
        args.data_path,
        f"{args.output_path}/brt",
        "--num_processes",
        str(args.process_num),
        "--no_random_name",
        "--method",
        "10",
        "--no_label",
    ]
)
