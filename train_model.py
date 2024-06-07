import pandas as pd
import torch
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')

# Inspect the dataset
print(df.head())

# Preprocess the data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

df['instruction'] = df['instruction'].apply(preprocess_text)
df['response'] = df['response'].apply(preprocess_text)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained('gpt2')

# Set the padding token
tokenizer.pad_token = tokenizer.eos_token

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

def tokenize_function(examples):
    return tokenizer(examples, padding='max_length', truncation=True, max_length=128)

# Tokenize the data
train_encodings = tokenize_function(train_df['instruction'].tolist())
train_labels = tokenize_function(train_df['response'].tolist())

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Create the dataset
train_dataset = CustomDataset(train_encodings, train_labels)

# Load the model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Set up the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')

print("Model training and saving complete.")
