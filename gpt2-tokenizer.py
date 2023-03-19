from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Smallest base model of GPT-2
model_name = "gpt2"

# Download the tokenizer and the model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
