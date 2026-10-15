
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()
print("Model loaded! Type 'exit' to quit.")


def ask_ai(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly voice assistant for elderly people. "
                "Use simple, clear language and short answers. "
                "Never invent medicine dosages or medical instructions. "
                "If information is uncertain, say so. "
                "For emergencies, advise contacting local emergency services."
            ),
        },
        {"role": "user", "content": question},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()



def main():
    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            print("\nAI:", ask_ai(question))
        except Exception as error:
            print("Error:", error)


if __name__ == "__main__":
    main()