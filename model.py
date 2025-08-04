from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline

model_name = "valhalla/t5-base-qg-hl"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

question_generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer
)

def generate_question_from_context(context: str) -> str:
    input_text = f"generate question: {context}"
    result = question_generator(input_text)
    return result[0]["generated_text"]
