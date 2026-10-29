"""Extract TRACE graphs from normalized image/prompt/response JSONL."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_builder import TRACEGraphBuilder


def add_model_arguments(parser):
    parser.add_argument("--model_type", default="llava-1.5-7b", choices=TRACEGraphBuilder.MODEL_CONFIGS)
    parser.add_argument("--model_path", help="Local weights directory or Hugging Face model ID")
    parser.add_argument("--cache_dir", help="Optional Hugging Face cache directory")
    parser.add_argument("--selected_layers", type=int, nargs="+", help="Zero-based indices; default: 8 equal-depth bin endpoints")
    parser.add_argument("--num_sampled_layers", type=int, default=8)
    parser.add_argument("--attn_threshold", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--load_in_4bit", action="store_true", help="Enable quantized backbone loading")
    quant.add_argument("--load_in_8bit", action="store_true", help="Enable quantized backbone loading")


def create_builder(args):
    return TRACEGraphBuilder(**{name: getattr(args, name) for name in (
        "model_type", "model_path", "cache_dir", "selected_layers", "num_sampled_layers",
        "attn_threshold", "top_k", "device", "load_in_4bit", "load_in_8bit",
    )})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--annotation_file", required=True, help="JSONL with image_path/image, prompt, response, binary label")
    parser.add_argument("--image_dir", default="", help="Root for relative image paths")
    parser.add_argument("--output_dir", required=True, help="Graph destination, e.g. data/graphs/train")
    parser.add_argument("--start_idx", type=int, default=0, help="Inclusive shard boundary")
    parser.add_argument("--end_idx", type=int, help="Exclusive shard boundary")
    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    builder = create_builder(args)
    builder.load_model()
    builder.process_dataset(args.annotation_file, args.output_dir, args.image_dir, args.start_idx, args.end_idx)


if __name__ == "__main__":
    main()
