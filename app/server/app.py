from flask import Flask,request,jsonify
from flask_cors import CORS
import pickle
import torch
import numpy as np
from utils import *
app=Flask(__name__)

# Enable CORS
CORS(app,resources={r"/translate": {"origins": "http://localhost:3000"}})

with open('./models/model-additive.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('./models/src_tokenizer.pkl', 'rb') as model_file:
    src_tokenizer = pickle.load(model_file)
with open('./models/trg_tokenizer.pkl', 'rb') as model_file:
    trg_tokenizer = pickle.load(model_file)

UNK_IDX = 0
PAD_IDX = 1
SOS_IDX = 2
EOS_IDX = 3
SPECIAL_TOKENS = ['<unk>', '<pad>', '<sos>', '<eos>']

def translate_sentence(model, sentence, src_tokenizer, trg_tokenizer, device, max_length=128):
    model.eval()
    
    # Tokenize and encode the source sentence
    src_tokens = torch.tensor([src_tokenizer.encode(sentence)]).to(device)
    
    # Initialize target sequence with <sos>
    trg_tokens = torch.tensor([[SOS_IDX]]).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get model prediction
            output, _ = model(src_tokens, trg_tokens)
            
            # Get the next token prediction
            pred_token = output.argmax(2)[:, -1].item()
            
            # Add predicted token to target sequence
            trg_tokens = torch.cat([trg_tokens, torch.tensor([[pred_token]]).to(device)], dim=1)
            
            # Stop if <eos> is predicted
            if pred_token == EOS_IDX:
                break
    
    # Convert tokens back to text
    translated_text = trg_tokenizer.decode(trg_tokens.squeeze().cpu().numpy())
    return translated_text
    
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/translate', methods=['GET'])
def translate():
    try:
        prompt =  request.args.get('prompt') 
        translation = translate_sentence(model, prompt, src_tokenizer, trg_tokenizer, get_device())
        # Perform the prediction using the loaded model
        
        return jsonify({'translation': translation})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/', methods=['GET'])
def call():
    return jsonify({'Name':"Suryansh Srivastava", 'ID':124997,'proglib':'NLP Assignment 3'})
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)