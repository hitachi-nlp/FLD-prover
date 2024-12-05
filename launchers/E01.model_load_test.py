from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("hitachi-nlp/Llama-3.1-70B-FLDx2")

# Load the model
model = AutoModelForCausalLM.from_pretrained("hitachi-nlp/Llama-3.1-70B-FLDx2")
