"""PyTorch dataset for cropped single-line OCR images."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from smartvision_ocr.data.validation import ImageTextSample, read_manifest


class OCRDataset(Dataset):
    """Loads validated one-line image/transcription samples."""

    def __init__(self, samples: list[ImageTextSample]) -> None:
        self.samples = samples

    @classmethod
    def from_csv(cls, csv_path: str | Path, data_dir: str | Path) -> "OCRDataset":
        return cls(read_manifest(csv_path, data_dir))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, str]:
        sample = self.samples[index]
        with Image.open(sample.image_path) as image:
            return image.convert("RGB"), sample.text
