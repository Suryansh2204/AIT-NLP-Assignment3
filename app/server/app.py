from flask import Flask,request,jsonify
from flask_cors import CORS
import pickle
import torch
import numpy as np
from LSTMLanguageModel import LSTMLanguageModel
app=Flask(__name__)

# Enable CORS
CORS(app,resources={r"/predict": {"origins": "http://localhost:3000"}})

with open('./models/LSTM_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('./models/tokenizer.pkl', 'rb') as model_file:
    tokenizer = pickle.load(model_file)
with open('./models/vocab.pkl', 'rb') as model_file:
    vocab = pickle.load(model_file)

def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            #prediction: [batch size, seq len, vocab size]
            #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']: #if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:    #if it is eos, we stop
                break

            indices.append(prediction) #autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens
    
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@app.route('/predict', methods=['GET'])
def predict():
    try:
        prompt =  request.args.get('prompt') 
        max_seq_len =  int(request.args.get('seqlen')) 
        
        # Perform the prediction using the loaded model
        seed = 0
        temperature = 0.6
        device=get_device()
        generation = generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed)
        return jsonify({'genText': generation})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/', methods=['GET'])
def call():
    return jsonify({'Name':"Suryansh Srivastava", 'ID':124997,'proglib':'NLP Assignment 2'})
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)