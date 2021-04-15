from flask import Flask, render_template, request
from flask import jsonify
#from Chatbot1 import response
from Chatbot2 import response
import pandas as pd
import numpy as np
import os
#df=pd.read_excel(r"C:\Users\91846\chatbot-master\chatsave.xlsx")

app = Flask(__name__)
@app.route("/")
def index(name=None):
    return render_template('index.html',name=name)

def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    userText=userText.lower()
    response1= str(response(userText))
    return jsonify(response1)
    #return response1
    
    
if __name__ == "__main__":
    app.run(host='0.0.0.0',  port = int(os.environ.get("PORT", 5000)) )