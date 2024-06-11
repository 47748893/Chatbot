import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_prepare_data(file_path):
    dataset = pd.read_csv(file_path)
    vectorizer = TfidfVectorizer().fit(dataset['instruction'])
    instruction_vectors = vectorizer.transform(dataset['instruction'])
    return dataset, vectorizer, instruction_vectors

def generate_response(user_query, vectorizer, instruction_vectors, dataset):
    query_vector = vectorizer.transform([user_query])
    similarities = cosine_similarity(query_vector, instruction_vectors).flatten()
    best_match_index = similarities.argmax()
    return dataset.iloc[best_match_index]['response']

def chatbot_main():
    file_path = 'Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
    dataset, vectorizer, instruction_vectors = load_and_prepare_data(file_path)
    
    print("Hello! How can I assist you today?")
    print("You can type 'exit' to end chat!")
    while True:
        user_query = input("You: ")
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye! Have a great day!")
            break
        response = generate_response(user_query, vectorizer, instruction_vectors, dataset)
        print(f"Chatbot: {response}")

if __name__ == "__main__":
    chatbot_main()