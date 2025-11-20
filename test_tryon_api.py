"""
Simple client script to exercise the IDM-VTON Modal API.

Usage examples:

Single garment:
    python test_tryon_api.py \
        --mode single \
        --human samples/human.png \
        --garment samples/garment.png \
        --output result.png

Batch (same person, multiple garments):
    python test_tryon_api.py \
        --mode batch \
        --human samples/human.png \
        --garment samples/garment1.png samples/garment2.png \
        --output batch_results.zip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import requests

BASE_URL = "https://vanshkhaneja2004--idm-vton-tryonmodel-api.modal.run"


def _validate_paths(paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for p in paths:
        rp = p.expanduser().resolve()
        if not rp.exists():
            raise FileNotFoundError(f"File does not exist: {rp}")
        resolved.append(rp)
    return resolved


def call_single_tryon(
    human_path: Path,
    garment_path: Path,
    output_path: Path,
    garment_description: str | None,
    auto_crop: bool,
) -> None:
    data: dict[str, str] = {}
    if garment_description:
        data["garment_description"] = garment_description
    if auto_crop:
        data["auto_crop"] = "true"

    with human_path.open("rb") as human_file, garment_path.open("rb") as garment_file:
        files = {
            "human_image": (human_path.name, human_file, "image/png"),
            "garment_image": (garment_path.name, garment_file, "image/png"),
        }
        response = requests.post(
            f"{BASE_URL}/tryon",
            files=files,
            data=data,
            timeout=600,
        )

    if response.status_code != 200:
        print("Request failed:", response.status_code, response.text)
        response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"Saved generated image to {output_path}")


def call_batch_tryon(
    human_path: Path,
    garment_paths: List[Path],
    output_path: Path,
    garment_descriptions: str | None,
    auto_crop: bool,
) -> None:
    data: dict[str, str] = {}
    if garment_descriptions:
        data["garment_descriptions"] = garment_descriptions
    if auto_crop:
        data["auto_crop"] = "true"

    files: List[tuple[str, tuple[str, object, str]]] = []
    human_file = human_path.open("rb")
    files.append(
        ("human_image", (human_path.name, human_file, "image/png")),
    )

    garment_file_handles = [p.open("rb") for p in garment_paths]
    try:
        for path, handle in zip(garment_paths, garment_file_handles):
            files.append(
                ("garment_images", (path.name, handle, "image/png")),
            )

        response = requests.post(
            f"{BASE_URL}/tryon/batch",
            files=files,
            data=data,
            timeout=1200,
        )
    finally:
        human_file.close()
        for handle in garment_file_handles:
            handle.close()

    if response.status_code != 200:
        print("Batch request failed:", response.status_code, response.text)
        response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    print(f"Saved batch results ZIP to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test IDM-VTON Modal API.")
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Select single garment or batch mode.",
    )
    parser.add_argument(
        "--human",
        type=Path,
        required=True,
        help="Path to the person image.",
    )
    parser.add_argument(
        "--garment",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to garment image(s).",
    )
    parser.add_argument(
        "--garment-description",
        type=str,
        default=None,
        help="Optional garment description text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tryon_result.png"),
        help="Output file path (.png for single, .zip for batch).",
    )
    parser.add_argument(
        "--batch-descriptions",
        type=str,
        default=None,
        help="Optional comma-separated descriptions for batch mode.",
    )
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        help="Enable YOLO-guided auto-cropping to preserve aspect ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    human_path = _validate_paths([args.human])[0]
    garment_paths = _validate_paths(args.garment)

    if args.mode == "single":
        if len(garment_paths) != 1:
            print("For single mode, provide exactly one garment image.", file=sys.stderr)
            sys.exit(1)
        call_single_tryon(
            human_path=human_path,
            garment_path=garment_paths[0],
            output_path=args.output,
            garment_description=args.garment_description,
            auto_crop=args.auto_crop,
        )
    else:
        call_batch_tryon(
            human_path=human_path,
            garment_paths=garment_paths,
            output_path=args.output,
            garment_descriptions=args.batch_descriptions,
            auto_crop=args.auto_crop,
        )


if __name__ == "__main__":
    main()

