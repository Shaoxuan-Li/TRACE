"""Normalize exported M-HalDetect and ViGoR records into TRACE JSONL.

Inputs are JSONL records with image_path/image, statement/response and either
is_hallucination or label, with one record per segment or sentence.
M-HalDetect categorical labels follow Appendix A.1:
Accurate/Analysis -> 0, Inaccurate/Unsure -> 1.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

PROMPT = "Please describe the content of this image in detail."
LABEL_MAP = {"accurate": 0, "analysis": 0, "inaccurate": 1, "unsure": 1}


def binary_label(value):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in LABEL_MAP:
            return LABEL_MAP[normalized]
        if normalized in ("true", "1"):
            return 1
        if normalized in ("false", "0"):
            return 0
    elif isinstance(value, (bool, int)) and value in (0, 1):
        return int(value)
    raise ValueError(f"Unsupported label {value!r}; expected binary or M-HalDetect category")


def convert(src_file, dst_file, min_words=0, image_dir="", check_images=False, prompt=PROMPT):
    """Validate and convert one exported split without implicit sample filtering."""
    if min_words < 0:
        raise ValueError("min_words must be nonnegative")
    stats = Counter()
    records = []
    with open(src_file, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            response = item.get("response", item.get("statement"))
            image = item.get("image_path", item.get("image"))
            if not isinstance(response, str) or not response.strip() or not isinstance(image, str) or not image:
                raise ValueError(f"{src_file}:{line_number}: missing nonempty image/response")
            if "is_hallucination" not in item and "label" not in item:
                raise ValueError(f"{src_file}:{line_number}: missing hallucination label")
            label = binary_label(item.get("is_hallucination", item.get("label")))
            stats["total"] += 1
            if len(response.split()) < min_words:
                stats["filtered_short"] += 1
                continue
            path = Path(image)
            if check_images and not (path if path.is_absolute() else Path(image_dir) / path).is_file():
                raise FileNotFoundError(f"{src_file}:{line_number}: {image}")
            record = {"image_path": image, "prompt": item.get("prompt", prompt), "response": response, "label": label}
            if "id" in item:
                record["id"] = item["id"]
            records.append(record)
            stats["hallucination" if label else "factual"] += 1
    if not records:
        raise ValueError(f"No usable records in {src_file}")
    destination = Path(dst_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats["written"] = len(records)
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhal_train", help="Exported M-HalDetect train JSONL")
    parser.add_argument("--mhal_val", help="Exported M-HalDetect validation JSONL")
    parser.add_argument("--vigor", help="Exported ViGoR JSONL, written as OOD test")
    parser.add_argument("--output_dir", default="data/annotations")
    parser.add_argument("--min_words", type=int, default=0, help="Minimum response word count (default: no length filtering)")
    parser.add_argument("--image_dir", default="")
    parser.add_argument("--check_images", action="store_true")
    parser.add_argument("--prompt", default=PROMPT, help="Fallback only when the exported record has no prompt")
    args = parser.parse_args()
    if not any((args.mhal_train, args.mhal_val, args.vigor)):
        parser.error("Provide at least one of --mhal_train, --mhal_val, --vigor")
    for split, source in (("train", args.mhal_train), ("val", args.mhal_val), ("test", args.vigor)):
        if source:
            stats = convert(source, Path(args.output_dir) / f"{split}.jsonl", args.min_words,
                            args.image_dir, args.check_images, args.prompt)
            print(f"{split}: {json.dumps(stats, sort_keys=True)}")


if __name__ == "__main__":
    main()
