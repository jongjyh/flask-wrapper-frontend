from flask import Flask, request 
import sys
import time
sys.path.append("..")

import torch
import pandas as pd
import numpy as np
import time
from transformers import BertTokenizer, BertForMaskedLM

# flask-swagger
from flask import Flask
from flask_restx import Api, Resource, fields,Namespace

app = Flask(__name__)
api = Api(app, version='0.1', title='Query Correction',
    description='debug some api',
)
ns = Namespace('test', description='test')
api.add_namespace(ns)
# loading model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("shibing624/macbert4csc-base-chinese")
model = BertForMaskedLM.from_pretrained("shibing624/macbert4csc-base-chinese")
model.to(device)

def inference(text,model,tokenizer):
    with torch.no_grad():
            start = time.time()
            outputs = model(**tokenizer(text, padding=True, return_tensors='pt').to(device))
            _text = tokenizer.decode(torch.argmax(outputs.logits, dim=-1).squeeze(0), skip_special_tokens=True).replace(' ', '')
            elapsed = (time.time() - start)
            corrected_text = _text[:len(text)]
    return corrected_text,elapsed

# get method
@ns.route('/pred/<string:query>', endpoint='query')
class Prediction(Resource):
    def get(self,query):
        text = query
        corrected_text,time = inference(text,model,tokenizer)
        data = {"raw_query":text,
            "fixed_query":corrected_text,
            "cost_time":round(time,5)}
        return data
query = ns.model('Query', {
    'query': fields.String(description='需要检查的句子', required=True)
})

# post method
@ns.route('/pred')
class Prediction(Resource):
    # @ns.marshal_with(query)
    @ns.expect(query) 
    def post(self):
        print(api.payload)
        params = request.json
        text = params['query']
        corrected_text,time = inference(text,model,tokenizer)
        data = {"raw_query":text,
            "fixed_query":corrected_text,
            "cost_time":round(time,5)}
        return data
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8890, debug=False)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
