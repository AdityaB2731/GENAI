# Install necessary libraries
!pip install transformers accelerate bitsandbytes

# Import
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load Phi-2 model + tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=True
)

# Get user input
question = input("Ask a question: ")

# Tokenize and generate
inputs = tokenizer(question, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)

# Decode and print
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nAnswer: {answer}")
in this where is peft squad bert and lora used and add some more feature here genai + search what is that
