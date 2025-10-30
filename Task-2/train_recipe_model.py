# train_recipe_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("json", data_files="recipes.jsonl")

# Combine prompt and response into one text field
def combine_texts(example):
    return {"text": example["prompt"] + " " + example["response"]}

dataset = dataset.map(combine_texts)

# Load tokenizer and model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a pad token if it doesn't exist (GPT-2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# Tell the model to use that pad token when padding sequences
model.config.pad_token_id = tokenizer.pad_token_id


# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)

# Set labels same as input_ids for causal language modeling
def add_labels(example):
    example["labels"] = example["input_ids"]
    return example

tokenized = tokenized.map(add_labels, batched=False)

tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# Split into train/test (80/20)
train_data = tokenized["train"].shuffle(seed=42).select(range(4))   # small sample
eval_data = tokenized["train"].shuffle(seed=42).select(range(1))   # mini eval

# Training arguments
args = TrainingArguments(
    output_dir="./recipe_model",
    overwrite_output_dir=True,
    num_train_epochs=8,
    per_device_train_batch_size=2,
    logging_steps=10,
    save_steps=100,
    fp16=False
)


# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_data,
    eval_dataset=eval_data
)

# Train!
trainer.train()

# Save fine-tuned model
trainer.save_model("./recipe_model")
tokenizer.save_pretrained("./recipe_model")
print("✅ Fine-tuning complete. Model saved to ./recipe_model")
