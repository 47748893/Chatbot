import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import pandas as pd
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def prepare_data(file_path, output_path, sample_frac=0.1):
    data = pd.read_csv(file_path)
    data['instruction'] = data['instruction'].apply(clean_text)
    data['response'] = data['response'].apply(clean_text)
    data.dropna(subset=['instruction', 'response'], inplace=True)
    data['input_output'] = data['instruction'] + ' ' + data['response']
    data_subset = data.sample(frac=sample_frac, random_state=42)
    
    with open(output_path, 'w') as f:
        for line in data_subset['input_output']:
            f.write(line + '\n')

def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128)

def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False)

def train_model(processed_data_path, model_save_path):
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    train_dataset = load_dataset(processed_data_path, tokenizer)
    data_collator = create_data_collator(tokenizer)
    
    model = GPT2LMHeadModel.from_pretrained('distilgpt2')

    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print("Fine-tuning complete and model saved.")

def generate_response(query, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(query, return_tensors='pt').to(model.device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def chat(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)

    print("Welcome to the Customer Service Chatbot. Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        response = generate_response(user_input, model, tokenizer)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    file_path = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
    processed_data_path = 'processed_data.txt'
    model_save_path = './fine_tuned_gpt2'

    # Prepare data
    prepare_data(file_path, processed_data_path)

    # Train model
    train_model(processed_data_path, model_save_path)

    # Start chat interface
    chat(model_save_path)
