"""Command-line validation and TrOCR crop export for canonical JSONL manifests."""

from __future__ import annotations

import argparse

from smartvision_ocr.data.manifest import export_trocr_split, validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a canonical JSONL manifest")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--image-root", required=True)

    export = commands.add_parser("export", help="export verified line crops for TrOCR")
    export.add_argument("--manifest", required=True)
    export.add_argument("--image-root", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--split", choices=("train", "validation", "test"), required=True)
    export.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "validate":
        report = validate_manifest(args.manifest, args.image_root)
        print(
            f"records={len(report.records)} verified={report.verified_count} "
            f"pseudo={report.pseudo_count} errors={len(report.errors)}"
        )
        for error in report.errors:
            print(f"ERROR: {error}")
        return 0 if report.valid else 1

    try:
        report = export_trocr_split(
            args.manifest,
            args.image_root,
            args.output_dir,
            args.split,
            overwrite=args.overwrite,
        )
    except (FileExistsError, ValueError, OSError) as error:
        print(f"ERROR: {error}")
        return 1
    excluded = ", ".join(f"{key}={value}" for key, value in sorted(report.excluded.items()))
    print(
        f"exported={report.exported} split={report.split} "
        f"csv={report.csv_path} provenance={report.provenance_path} "
        f"excluded={excluded or 'none'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
