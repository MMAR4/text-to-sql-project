from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(model_name)

model = PeftModel.from_pretrained(base_model, "text_to_sql_lora_final")

def generate_sql(question):
    prompt = f"Question: {question}\nSQL:"
    
    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)