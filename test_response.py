import torch
from transformers import GPT2LMHeadModel, AutoTokenizer


# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./model')
tokenizer = AutoTokenizer.from_pretrained('./model')

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")

def generate_response(user_query, max_length=50, num_beams=5, no_repeat_ngram_size=2):
    """Generates a response using beam search with repetition penalty."""
    
    print("User Query:", user_query)
    input_ids = tokenizer.encode(user_query, return_tensors='pt').to(device)

    outputs = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,   # Now this should work after updating!
        early_stopping=True,
        do_sample=True,       
        top_k=50,              
        top_p=0.95,              
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated Response:", response)
    return response



# Example user prompt query
user_query = "I have a question about my order"
generated_response = generate_response(user_query)
print("Generated Response:", generated_response)
