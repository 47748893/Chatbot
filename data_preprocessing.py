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

print("Pre-processing done.")

if __name__ == "__main__":
    file_path = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
    output_path = 'processed_data.txt'
    prepare_data(file_path, output_path)
