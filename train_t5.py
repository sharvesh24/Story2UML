from transformers import T5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from utils import load_dataset_from_docx
import torch

examples = load_dataset_from_docx("User_Stories.docx")
train_data, test_data = train_test_split(examples, test_size=0.1, random_state=42)

train_dataset = Dataset.from_list(train_data)
test_dataset = Dataset.from_list(test_data)

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize(batch):
    input_enc = tokenizer(batch["input"], padding="max_length", truncation=True, max_length=512)
    target_enc = tokenizer(batch["output"], padding="max_length", truncation=True, max_length=512)
    input_enc["labels"] = target_enc["input_ids"]
    return input_enc

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

args = Seq2SeqTrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    save_strategy="epoch",
    predict_with_generate=True,
    logging_dir="./logs",
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()
trainer.save_model("./model")
tokenizer.save_pretrained("./model")

output_dir = "./model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"✅ Model and tokenizer saved to: {output_dir}")
