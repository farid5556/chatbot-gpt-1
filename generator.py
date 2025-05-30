from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load IndoT5 QA Model
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_answer(context, question):
    input_text = f"question: {question}  context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(input_ids, max_length=256)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
