!rm -rf ~/.cache/huggingface/datasets/ag_news
# 🚀 STEP 1: Install dependencies
!pip install transformers peft datasets accelerate bitsandbytes

# 🚀 STEP 2: Import libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# 🚀 STEP 3: Load model + tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto"
)

# 🚀 STEP 4: Prepare model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 🚀 STEP 5: Load & tokenize SQuAD dataset (small slice for demo)
dataset = load_dataset("ag_news", split="train[:0.5%]")

def tokenize_function(example):
    return tokenizer(
        f"Question: {example['question']} Context: {example['context']} Answer: {example['answers']['text'][0]}",
        truncation=True
    )
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 🚀 STEP 6: Set up training
training_args = TrainingArguments(
    output_dir="./asked_model",
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=1,
    save_steps=10,
    save_total_limit=2
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 🚀 STEP 7: Fine-tune the model
trainer.train()

# 🚀 STEP 8: Save the fine-tuned model
model.save_pretrained("./asked_model")
tokenizer.save_pretrained("./asked_model")

# 🚀 STEP 9: Load and chat with the fine-tuned model
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./asked_model")
model = AutoModelForCausalLM.from_pretrained("./asked_model", device_map="auto")

while True:
    question = input("Ask a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nAnswer: {answer}\n")
