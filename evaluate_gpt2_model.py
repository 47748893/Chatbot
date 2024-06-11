import re
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer

def load_dataset(file_path, tokenizer):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128)

def create_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

def prepare_eval_data(file_path, output_path, sample_frac=0.1):
    data = pd.read_csv(file_path)
    data['instruction'] = data['instruction'].apply(clean_text)
    data['response'] = data['response'].apply(clean_text)
    data.dropna(subset=['instruction', 'response'], inplace=True)
    data['input_output'] = data['instruction'] + ' ' + data['response']
    data_subset = data.sample(frac=sample_frac, random_state=42)
    
    with open(output_path, 'w') as f:
        for line in data_subset['input_output']:
            f.write(line + '\n')

def evaluate_model(eval_data_path, model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    eval_dataset = load_dataset(eval_data_path, tokenizer)
    data_collator = create_data_collator(tokenizer)
    
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        eval_dataset=eval_dataset,
    )
    
    eval_results = trainer.evaluate()
    eval_perplexity = torch.exp(torch.tensor(eval_results['eval_loss']))

    print("Evaluation Results:", eval_results)
    print("Evaluation Perplexity:", eval_perplexity.item())

if __name__ == "__main__":
    eval_file_path = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
    eval_processed_data_path = 'eval_processed_data.txt'
    prepare_eval_data(eval_file_path, eval_processed_data_path)
    model_path = './fine_tuned_gpt2'
    evaluate_model(eval_processed_data_path, model_path)
