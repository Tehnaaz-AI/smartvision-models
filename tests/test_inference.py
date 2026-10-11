from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from smartvision_ocr.inference import predict


class FakeProcessor:
    def __init__(self):
        self.image_mode = None

    def __call__(self, *, images, return_tensors):
        self.image_mode = images.mode
        return SimpleNamespace(pixel_values=torch.zeros((1, 3, 4, 4)))

    def batch_decode(self, generated_ids, skip_special_tokens):
        assert skip_special_tokens is True
        return ["Paracetamol 500 mg"]


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(
            decoder_start_token_id=2,
            pad_token_id=1,
            eos_token_id=2,
        )
        self.device = None
        self.is_eval = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.is_eval = True
        return self

    def generate(self, pixel_values):
        assert torch.is_inference_mode_enabled()
        assert pixel_values.device.type == "cpu"
        return torch.tensor([[1, 2]])


def test_predict_image_converts_to_rgb_and_decodes_on_cpu(tmp_path, monkeypatch):
    image_path = tmp_path / "line.png"
    Image.new("L", (20, 10), color=255).save(image_path)
    model = FakeModel()
    processor = FakeProcessor()
    monkeypatch.setattr(predict.torch.cuda, "is_available", lambda: False)

    text = predict.predict_image(
        image_path,
        model=model,
        processor=processor,
    )

    assert text == "Paracetamol 500 mg"
    assert processor.image_mode == "RGB"
    assert model.device == torch.device("cpu")
    assert model.is_eval


def test_predict_image_reports_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="Input image does not exist"):
        predict.predict_image(tmp_path / "missing.png")


def test_predict_image_reports_invalid_image(tmp_path):
    image_path = tmp_path / "not-an-image.png"
    image_path.write_text("not a png", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read image"):
        predict.predict_image(image_path)


def test_load_model_reports_invalid_generation_configuration(monkeypatch):
    class FakeProcessorLoader:
        @staticmethod
        def from_pretrained(model_id):
            return object()

    class FakeModelLoader:
        @staticmethod
        def from_pretrained(model_id):
            return SimpleNamespace(
                generation_config=SimpleNamespace(pad_token_id=1)
            )

    monkeypatch.setattr(predict, "TrOCRProcessor", FakeProcessorLoader)
    monkeypatch.setattr(predict, "VisionEncoderDecoderModel", FakeModelLoader)

    with pytest.raises(ValueError, match="invalid generation configuration"):
        predict.load_model("test-model")


def test_load_model_reports_download_failure(monkeypatch):
    class FailingProcessorLoader:
        @staticmethod
        def from_pretrained(model_id):
            raise OSError("network unavailable")

    monkeypatch.setattr(predict, "TrOCRProcessor", FailingProcessorLoader)

    with pytest.raises(RuntimeError, match="Check the model name, network connection"):
        predict.load_model("test-model")


def test_cli_prints_model_device_image_and_prediction(monkeypatch, tmp_path, capsys):
    image_path = tmp_path / "line.png"
    monkeypatch.setattr(
        predict,
        "select_device",
        lambda requested=None: torch.device("cpu"),
    )
    monkeypatch.setattr(
        predict,
        "predict_image",
        lambda *args, **kwargs: "Paracetamol 500 mg",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "predict",
            str(image_path),
            "--model",
            "microsoft/trocr-base-printed",
        ],
    )

    predict.main()

    output = capsys.readouterr().out
    assert "Model: microsoft/trocr-base-printed" in output
    assert "Device: cpu" in output
    assert f"Input image: {image_path}" in output
    assert "Predicted text: Paracetamol 500 mg" in output
