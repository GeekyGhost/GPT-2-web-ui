from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)  # device=0 means using your GPU
