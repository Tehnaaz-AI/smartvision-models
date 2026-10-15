"""Interactively crop printed text lines for manual OCR evaluation."""

import argparse
import csv
import hashlib
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = PROJECT_ROOT / "data" / "evaluation" / "candidates.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "images"
CANDIDATE_FIELDS = ("source_image", "split", "review_note")
SUPPORTED_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
CROP_SOURCE_FIELDS = (
    "crop",
    "source_image",
    "split",
    "left",
    "top",
    "right",
    "bottom",
)


@dataclass(frozen=True)
class CropCandidate:
    source_image: str
    source_path: Path
    split: str
    review_note: str


def load_candidates(
    candidate_csv: str | Path,
    project_root: str | Path = PROJECT_ROOT,
) -> list[CropCandidate]:
    manifest_path = Path(candidate_csv).expanduser()
    root = Path(project_root).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Candidate CSV does not exist: {manifest_path}")

    candidates: list[CropCandidate] = []
    seen_images: set[Path] = set()
    seen_image_digests: set[str] = set()
    splits: set[str] = set()
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file, strict=True)
            if reader.fieldnames != list(CANDIDATE_FIELDS):
                raise ValueError(
                    f"{manifest_path} must contain exactly these columns: "
                    f"{','.join(CANDIDATE_FIELDS)}."
                )
            for row_number, row in enumerate(reader, start=2):
                source_image = (row.get("source_image") or "").strip()
                split = (row.get("split") or "").strip()
                review_note = (row.get("review_note") or "").strip()
                if None in row:
                    raise ValueError(
                        f"Candidate CSV row {row_number} has more values than columns."
                    )
                if not source_image or not split:
                    raise ValueError(
                        f"Candidate CSV row {row_number} needs source_image and split."
                    )
                source_ref = Path(source_image)
                if source_ref.is_absolute():
                    raise ValueError(
                        f"Candidate image paths must be relative to the project: "
                        f"{source_image}"
                    )
                source_path = (root / source_ref).resolve()
                if not source_path.is_relative_to(root):
                    raise ValueError(
                        f"Candidate image path escapes the project directory: "
                        f"{source_image}"
                    )
                if split.casefold() not in {
                    part.casefold() for part in source_ref.parts
                }:
                    raise ValueError(
                        f"Candidate row {row_number} declares split '{split}', "
                        f"but that directory is not in source_image: {source_image}"
                    )
                if source_ref.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
                    raise ValueError(
                        f"Candidate image from row {row_number} has an unsupported "
                        f"image extension: {source_image}"
                    )
                if source_path in seen_images:
                    raise ValueError(
                        f"Candidate image is listed more than once: {source_image}"
                    )
                if not source_path.is_file():
                    raise FileNotFoundError(
                        f"Candidate image from row {row_number} does not exist: "
                        f"{source_path}"
                    )
                try:
                    with Image.open(source_path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError) as error:
                    raise ValueError(
                        f"Candidate image from row {row_number} is unreadable: "
                        f"{source_path}: {error}"
                    ) from error
                image_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
                if image_digest in seen_image_digests:
                    raise ValueError(
                        f"Candidate image from row {row_number} duplicates an image "
                        f"already in the candidate list: {source_image}"
                    )

                candidates.append(
                    CropCandidate(
                        source_image=source_image,
                        source_path=source_path,
                        split=split,
                        review_note=review_note,
                    )
                )
                seen_images.add(source_path)
                seen_image_digests.add(image_digest)
                splits.add(split)
    except csv.Error as error:
        raise ValueError(f"Could not parse candidate CSV {manifest_path}: {error}") from error

    if not candidates:
        raise ValueError(f"No image candidates found in {manifest_path}.")
    if len(splits) != 1:
        raise ValueError(
            "The candidate list must use one dataset split only; do not mix "
            "training and held-out images."
        )
    return candidates


def crop_bounds(
    start: tuple[int, int],
    end: tuple[int, int],
    image_origin: tuple[int, int],
    scale: tuple[float, float],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Convert a canvas selection to clamped original-image pixel bounds."""
    left = math.floor((min(start[0], end[0]) - image_origin[0]) / scale[0])
    top = math.floor((min(start[1], end[1]) - image_origin[1]) / scale[1])
    right = math.ceil((max(start[0], end[0]) - image_origin[0]) / scale[0])
    bottom = math.ceil((max(start[1], end[1]) - image_origin[1]) / scale[1])
    width, height = image_size
    left = max(0, min(left, width))
    top = max(0, min(top, height))
    right = max(0, min(right, width))
    bottom = max(0, min(bottom, height))
    if right - left < 2 or bottom - top < 2:
        raise ValueError("Select a crop at least 2 pixels wide and high.")
    return left, top, right, bottom


def fit_preview_size(
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> tuple[int, int]:
    """Fit an image within a canvas without enlarging its original pixels."""
    image_width, image_height = image_size
    canvas_width, canvas_height = canvas_size
    if image_width < 1 or image_height < 1:
        raise ValueError("Image dimensions must be positive.")

    scale = min(
        1.0,
        max(1, canvas_width) / image_width,
        max(1, canvas_height) / image_height,
    )
    return (
        max(1, round(image_width * scale)),
        max(1, round(image_height * scale)),
    )


def save_selected_crop(
    source_image: Image.Image,
    bounds: tuple[int, int, int, int],
    output_dir: str | Path,
    source_path: str | Path,
    split: str,
    project_root: str | Path = PROJECT_ROOT,
) -> Path:
    """Save a lossless crop and atomically record its source image and bounds."""
    output = Path(output_dir).expanduser().resolve()
    root = Path(project_root).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    if not output.is_relative_to(root):
        raise ValueError(f"Crop output directory is outside the project: {output}")
    if not source.is_relative_to(root):
        raise ValueError(f"Source image is outside the project directory: {source}")
    if not split.strip():
        raise ValueError("Dataset split must not be empty.")
    left, top, right, bottom = bounds
    if (
        left < 0
        or top < 0
        or right > source_image.width
        or bottom > source_image.height
        or right - left < 2
        or bottom - top < 2
    ):
        raise ValueError(
            f"Crop bounds {bounds} are invalid for source image size "
            f"{source_image.size}."
        )

    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output.parent / "crop_sources.csv"
    provenance_rows: list[dict[str, str]] = []
    if provenance_path.exists():
        try:
            with provenance_path.open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:
                reader = csv.DictReader(file, strict=True)
                if reader.fieldnames != list(CROP_SOURCE_FIELDS):
                    raise ValueError(
                        f"{provenance_path} must contain exactly these columns: "
                        f"{','.join(CROP_SOURCE_FIELDS)}."
                    )
                for row_number, row in enumerate(reader, start=2):
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError(
                            f"{provenance_path}:{row_number}: malformed crop source row."
                        )
                    provenance_rows.append(row)
        except csv.Error as error:
            raise ValueError(
                f"Could not parse crop source manifest {provenance_path}: {error}"
            ) from error

    used_names = {
        row["crop"]
        for row in provenance_rows
        if row.get("crop")
    }
    number = 1
    while (
        f"line_{number:03}.png" in used_names
        or (output / f"line_{number:03}.png").exists()
    ):
        number += 1
    crop_name = f"line_{number:03}.png"
    crop_path = output / crop_name
    row = {
        "crop": crop_path.relative_to(root).as_posix(),
        "source_image": source.relative_to(root).as_posix(),
        "split": split,
        "left": str(bounds[0]),
        "top": str(bounds[1]),
        "right": str(bounds[2]),
        "bottom": str(bounds[3]),
    }
    provenance_rows.append(row)
    image_buffer = io.BytesIO()
    source_image.crop(bounds).save(image_buffer, format="PNG")

    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    temp_image_path: Path | None = None
    temp_manifest_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output, suffix=".tmp", delete=False
        ) as temporary_image:
            temp_image_path = Path(temporary_image.name)
            temporary_image.write(image_buffer.getvalue())

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=provenance_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_manifest:
            temp_manifest_path = Path(temporary_manifest.name)
            writer = csv.DictWriter(
                temporary_manifest, fieldnames=CROP_SOURCE_FIELDS
            )
            writer.writeheader()
            writer.writerows(provenance_rows)

        os.replace(temp_image_path, crop_path)
        temp_image_path = None
        try:
            os.replace(temp_manifest_path, provenance_path)
            temp_manifest_path = None
        except OSError:
            crop_path.unlink()
            raise
    finally:
        if temp_image_path is not None:
            temp_image_path.unlink(missing_ok=True)
        if temp_manifest_path is not None:
            temp_manifest_path.unlink(missing_ok=True)
    return crop_path


class EvaluationCropper:
    def __init__(
        self,
        root: tk.Tk,
        candidates: list[CropCandidate],
        output_dir: Path,
    ) -> None:
        self.root = root
        self.candidates = candidates
        self.output_dir = output_dir
        self.index = 0
        self.source_image: Image.Image | None = None
        self.preview_image: ImageTk.PhotoImage | None = None
        self.image_origin = (0, 0)
        self.scale = (1.0, 1.0)
        self.drag_start: tuple[int, int] | None = None
        self.selection: tuple[int, int, int, int] | None = None
        self.selection_id: int | None = None

        root.title("SmartVision OCR - manual evaluation crop selection")
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        window_width = min(1100, max(640, screen_width - 80))
        window_height = min(820, max(500, screen_height - 100))
        root.geometry(f"{window_width}x{window_height}")
        root.minsize(min(640, window_width), min(480, window_height))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)

        canvas_frame = tk.Frame(root, padx=10, pady=10)
        canvas_frame.grid(row=0, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, background="#333333")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        self.status = tk.StringVar()
        self.status_label = tk.Label(
            root, textvariable=self.status, anchor="w", justify="left", padx=10
        )
        self.status_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))
        self.status_label.bind(
            "<Configure>",
            lambda event: self.status_label.configure(
                wraplength=max(300, event.width - 20)
            ),
        )

        controls = tk.Frame(root, padx=8, pady=8)
        controls.grid(row=2, column=0, sticky="ew")
        for column in range(5):
            controls.columnconfigure(column, weight=1)
        button_options = {"padx": 4, "pady": 4, "sticky": "ew"}
        tk.Button(controls, text="Previous", command=self.previous).grid(
            row=0, column=0, **button_options
        )
        tk.Button(controls, text="Next", command=self.next).grid(
            row=0, column=1, **button_options
        )
        tk.Button(
            controls, text="Clear selection", command=self.clear_selection
        ).grid(row=0, column=2, **button_options)
        tk.Button(controls, text="Save crop", command=self.save_crop).grid(
            row=0, column=3, **button_options
        )
        tk.Button(controls, text="Close", command=root.destroy).grid(
            row=0, column=4, **button_options
        )
        self.canvas.bind("<ButtonPress-1>", self.start_selection)
        self.canvas.bind("<B1-Motion>", self.update_selection)
        self.canvas.bind("<ButtonRelease-1>", self.finish_selection)
        self.show_candidate()

    def _on_canvas_resize(self, _event: tk.Event) -> None:
        if self.source_image is not None:
            self.selection = None
            self.drag_start = None
            self._draw_preview()

    def show_candidate(self) -> None:
        candidate = self.candidates[self.index]
        with Image.open(candidate.source_path) as image:
            self.source_image = image.convert("RGB")

        self._draw_preview()
        self.selection = None
        self.selection_id = None
        self.status.set(
            f"Candidate {self.index + 1}/{len(self.candidates)} | "
            f"{candidate.split} | {candidate.source_image}\n"
            f"{candidate.review_note}\n"
            "Drag around exactly one fully visible text line; crops use original pixels."
        )

    def _draw_preview(self) -> None:
        if self.source_image is None:
            return

        canvas_size = (self.canvas.winfo_width(), self.canvas.winfo_height())
        preview_size = fit_preview_size(self.source_image.size, canvas_size)
        preview = self.source_image.copy()
        preview.thumbnail(preview_size, Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(preview)
        self.scale = (
            self.source_image.width / preview.width,
            self.source_image.height / preview.height,
        )
        self.image_origin = (
            (canvas_size[0] - preview.width) // 2,
            (canvas_size[1] - preview.height) // 2,
        )
        self.canvas.delete("all")
        self.canvas.create_image(
            self.image_origin[0],
            self.image_origin[1],
            anchor="nw",
            image=self.preview_image,
        )

    def start_selection(self, event: tk.Event) -> None:
        self.drag_start = (event.x, event.y)
        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
            self.selection_id = None

    def update_selection(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
        self.selection_id = self.canvas.create_rectangle(
            *self.drag_start,
            event.x,
            event.y,
            outline="#00ff00",
            width=2,
        )

    def finish_selection(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        self.selection = (*self.drag_start, event.x, event.y)
        self.drag_start = None

    def clear_selection(self) -> None:
        self.selection = None
        if self.selection_id is not None:
            self.canvas.delete(self.selection_id)
            self.selection_id = None

    def save_crop(self) -> None:
        if self.source_image is None or self.selection is None:
            messagebox.showerror("No crop selected", "Drag to select a text line first.")
            return
        try:
            bounds = crop_bounds(
                self.selection[:2],
                self.selection[2:],
                self.image_origin,
                self.scale,
                self.source_image.size,
            )
        except ValueError as error:
            messagebox.showerror("Invalid crop", str(error))
            return

        candidate = self.candidates[self.index]
        try:
            output_path = save_selected_crop(
                self.source_image,
                bounds,
                self.output_dir,
                candidate.source_path,
                candidate.split,
            )
        except (OSError, ValueError, csv.Error) as error:
            messagebox.showerror("Could not save crop", str(error))
            return
        messagebox.showinfo(
            "Crop saved",
            f"Saved {output_path.name} and recorded its source in "
            "data\\evaluation\\crop_sources.csv.\n"
            "Manually inspect it, then add its exact text to references.csv.",
        )
        self.clear_selection()

    def previous(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.show_candidate()

    def next(self) -> None:
        if self.index + 1 < len(self.candidates):
            self.index += 1
            self.show_candidate()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select and save image crops for manual OCR evaluation. "
            "This tool does not run OCR or create reference labels."
        )
    )
    parser.add_argument(
        "--candidate-csv",
        default=str(DEFAULT_CANDIDATES),
        help="CSV of human-review candidate image paths.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for selected PNG crops.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        candidates = load_candidates(args.candidate_csv)
    except (FileNotFoundError, OSError, ValueError) as error:
        parser.error(str(error))
    root = tk.Tk()
    EvaluationCropper(root, candidates, Path(args.output_dir).expanduser())
    root.mainloop()


if __name__ == "__main__":
    main()
