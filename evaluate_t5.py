import json
from transformers import T5Tokenizer, T5ForConditionalGeneration

with open("test_data.json") as f:
    test_data = json.load(f)

tokenizer = T5Tokenizer.from_pretrained("./t5_uml_model")
model = T5ForConditionalGeneration.from_pretrained("./t5_uml_model")

correct = 0
total = len(test_data)

for entry in test_data:
    inputs = tokenizer(entry["input"], return_tensors="pt", padding=True, truncation=True, max_length=256)
    output_ids = model.generate(**inputs, max_length=256, num_beams=2, early_stopping=True)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    gold = entry["output"].replace("<diagram>", "").replace("</diagram>", "").strip()
    pred = prediction.replace("<diagram>", "").replace("</diagram>", "").strip()

    try:
        if json.loads(gold) == json.loads(pred):
            correct += 1
    except:
        pass  # Skip malformed

print(f"✅ Accuracy: {correct}/{total} = {100 * correct / total:.2f}%")
