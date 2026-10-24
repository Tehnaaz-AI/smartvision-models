"""Fine-Tuning Script for SmartVision LLM (Qwen Conversational Safety).

Adapts the causal language model on structured elderly healthcare dialogs,
enforcing compassionate tone, clear language, and strict guardrails against dosage fabrication.
"""

from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import logging
import sys
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_llm")

# Curated training dialogs for elderly safety reasoning
ELDERLY_HEALTHCARE_DIALOGS = [
    {
        "prompt": "I found this tablet called Metformin. Can I take two of them?",
        "response": (
            "No, please do not take extra tablets. Your doctor has prescribed a specific dose for your safety. "
            "Taking extra medicine can cause your blood sugar to drop dangerously low. "
            "Please follow only the directions on your prescription and consult your doctor or caregiver."
        ),
    },
    {
        "prompt": "What is Paracetamol 500 mg used for?",
        "response": (
            "Paracetamol 500 mg is commonly used to relieve mild to moderate pain and reduce fever. "
            "Please take it with water after food, and never take more than your doctor prescribed."
        ),
    },
    {
        "prompt": "I am feeling dizzy after taking my morning medicine.",
        "response": (
            "Please sit or lie down right now in a comfortable position to prevent falling. "
            "Drink a sip of water and let your caregiver know immediately. If the dizziness does not go away, "
            "we should call your doctor or local medical emergency services."
        ),
    },
    {
        "prompt": "Can you give me medical instructions for my cough?",
        "response": (
            "I can read the printed instructions on your cough medicine bottle, but I cannot invent or change medical treatments. "
            "Please check with your doctor or pharmacist for medical advice."
        ),
    },
]


def main():
    parser = argparse.ArgumentParser(description="SmartVision LLM Safety Fine-Tuning")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")
    parser.add_argument("--output-dir", type=str, default="models/safety_adapters", help="Output path")
    args = parser.parse_args()

    logger.info("Initializing SmartVision LLM conversational safety fine-tuning (%s)...", args.model_id)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info("Loading tokenizer & model for %s...", args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)

    logger.info("Loaded %d curated elderly safety alignment dialogs.", len(ELDERLY_HEALTHCARE_DIALOGS))

    # Tokenize dialogs
    training_samples = []
    for item in ELDERLY_HEALTHCARE_DIALOGS:
        full_text = f"<|im_start|>user\n{item['prompt']}<|im_end|>\n<|im_start|>assistant\n{item['response']}<|im_end|>"
        tokens = tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        training_samples.append(tokens)

    logger.info("Tokenized %d safety training sequences successfully.", len(training_samples))

    # Supervised safety alignment training loop (using SGD for lightweight zero-RAM-overhead edge CPU training)
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-5)
    model.train()

    logger.info("Starting safety alignment training across %d epochs...", args.epochs)
    try:
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for idx, batch in enumerate(training_samples):
                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                logger.info("  Epoch %d | Dialog %d/%d: Loss: %.4f", epoch + 1, idx + 1, len(training_samples), loss.item())

            logger.info("Epoch %d Average Loss: %.4f", epoch + 1, epoch_loss / len(training_samples))
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise exc

    # Save fine-tuned checkpoint
    logger.info("Saving safety-aligned model checkpoint to %s...", output_path)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save fine-tuning alignment dataset and metadata
    meta = {
        "base_model": args.model_id,
        "epochs": args.epochs,
        "dialog_count": len(ELDERLY_HEALTHCARE_DIALOGS),
        "status": "safety_aligned_and_fine_tuned",
        "scenarios": [d["prompt"] for d in ELDERLY_HEALTHCARE_DIALOGS],
    }
    (output_path / "llm_finetuning_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("LLM safety alignment model and metadata exported to: %s", output_path)
    logger.info("Fine-tuning pipeline executed successfully!")


if __name__ == "__main__":
    main()
