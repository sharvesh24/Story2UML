from docx import Document
import re
import json

def load_dataset_from_docx(path):
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    pairs = re.split(r"\n(?=As a )", text)

    dataset = []
    for pair in pairs:
        match = re.match(r"^(As .*?)\n({.*})", pair, re.DOTALL)
        if match:
            input_text = match.group(1).strip()
            output_json = match.group(2).strip().replace('\n', '')
            try:
                json.loads(output_json)  # Validate JSON
                dataset.append({
                    "input": input_text,
                    "output": f"<diagram>{output_json}</diagram>"
                })
            except json.JSONDecodeError:
                continue
    return dataset
