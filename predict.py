import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset, Dataset
import torch

model_dir = "./model"
model = T5ForConditionalGeneration.from_pretrained(model_dir)
tokenizer = T5Tokenizer.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


with open("uml_training_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

test_data = [d for d in data if d.get("split") == "test"]
if not test_data:
    raise ValueError("No test examples found.")

correct = 0
total = len(test_data)

print(f"📊 Evaluating on {total} test examples...")

for i, example in enumerate(test_data):
    input_text = example["input"]
    true_output = example["output"]

    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        generated_ids = model.generate(input_ids, max_length=512)
    predicted = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    if predicted == true_output.strip():
        correct += 1

   
    print(f"\n🔍 Example {i+1}")
    print(f"Input: {input_text}")
    print(f"✅ Prediction: {predicted}")
    print(f"🎯 Ground Truth: {true_output}")
    print(f"{'✔️ Match' if predicted == true_output.strip() else '❌ Mismatch'}")

accuracy = (correct / total) * 100
print(f"\n✅ Accuracy: {accuracy:.2f}% ({correct}/{total} correct)")
