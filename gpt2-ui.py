import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_text(prompt):
    set_seed(42)
    generated = generator(prompt, max_length=250, num_return_sequences=1)
    return generated[0]["generated_text"]

input_text = gr.inputs.Textbox(lines=3, placeholder="Enter a text prompt...")
output_text = gr.outputs.Textbox()

iface = gr.Interface(
    fn=generate_text, inputs=input_text, outputs=output_text, title="GPT-2 Text Generation"
)
iface.launch()

