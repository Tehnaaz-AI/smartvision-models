"""Validation for image/text CSV manifests and leakage-safe splits."""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


@dataclass(frozen=True)
class ImageTextSample:
    image_path: Path
    text: str


def read_manifest(csv_path: str | Path, data_dir: str | Path) -> list[ImageTextSample]:
    """Read and validate a UTF-8 CSV containing relative image paths and labels."""
    manifest_path = Path(csv_path)
    root = Path(data_dir).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CSV manifest does not exist: {manifest_path}")

    samples: list[ImageTextSample] = []
    seen_images: set[Path] = set()
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            if (
                reader.fieldnames is None
                or len(reader.fieldnames) != 2
                or set(reader.fieldnames) != {"image", "text"}
            ):
                raise ValueError(
                    f"{manifest_path} must contain CSV columns named 'image' and 'text'."
                )

            for row_number, row in enumerate(reader, start=2):
                image_value = (row.get("image") or "").strip()
                text_value = row.get("text")
                if not image_value:
                    raise ValueError(f"{manifest_path}:{row_number}: image path is empty.")
                if text_value is None or not text_value.strip():
                    raise ValueError(f"{manifest_path}:{row_number}: text label is empty.")

                relative_path = Path(image_value)
                if relative_path.is_absolute():
                    raise ValueError(
                        f"{manifest_path}:{row_number}: image path must be relative to "
                        f"the dataset directory: {image_value}"
                    )
                if relative_path.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
                    raise ValueError(
                        f"{manifest_path}:{row_number}: unsupported image file "
                        f"extension: {relative_path.suffix or '(none)'}."
                    )
                image_path = (root / relative_path).resolve()
                if not image_path.is_relative_to(root):
                    raise ValueError(
                        f"{manifest_path}:{row_number}: image path escapes the dataset "
                        f"directory: {image_value}"
                    )
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"{manifest_path}:{row_number}: image does not exist: {image_path}"
                    )
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    raise ValueError(
                        f"{manifest_path}:{row_number}: unreadable image "
                        f"{image_path}: {error}"
                    ) from error

                if image_path in seen_images:
                    raise ValueError(
                        f"{manifest_path}:{row_number}: image is referenced more "
                        f"than once: {image_value}"
                    )
                seen_images.add(image_path)
                samples.append(ImageTextSample(image_path=image_path, text=text_value))
    except csv.Error as error:
        raise ValueError(f"Could not read CSV manifest {manifest_path}: {error}") from error

    if not samples:
        raise ValueError(f"CSV manifest contains no data rows: {manifest_path}")
    return samples


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_disjoint_splits(
    splits: dict[str, list[ImageTextSample]],
) -> None:
    """Reject shared image paths and byte-identical images across data splits."""
    seen_paths: dict[Path, str] = {}
    seen_digests: dict[str, tuple[str, Path]] = {}
    for split_name, samples in splits.items():
        for sample in samples:
            path = sample.image_path.resolve()
            previous_split = seen_paths.get(path)
            if previous_split is not None:
                raise ValueError(
                    f"Data leakage: image {path} appears in both "
                    f"'{previous_split}' and '{split_name}' splits."
                )
            seen_paths[path] = split_name

            digest = _sha256(path)
            previous = seen_digests.get(digest)
            if previous is not None:
                previous_split, previous_path = previous
                raise ValueError(
                    f"Data leakage: byte-identical images appear in "
                    f"'{previous_split}' ({previous_path}) and '{split_name}' ({path})."
                )
            seen_digests[digest] = (split_name, path)
