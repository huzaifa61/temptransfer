import os
import sys

from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import cv2
import tensorflow as tf
from tensorflow import keras
from sklearn import shuffle
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


import numpy as np
from util import base64_to_pil

app = Flask(__name__)
app.config['UPLOAD FOLDER']='predict/'
app.config['ALLOWED_EXTENSIONS']=set(['png', 'jpg', 'jpeg'])



print('Model loaded. Check http://127.0.0.1:5000/')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

MODEL_PATH = 'models/final_classification.h5'


model = load_model(MODEL_PATH)
model._make_predict_function()
print('Model loaded. Start serving...')

def get_images(directory):
	Images,Labels=[],[]
	label=0
	for labels in os.listdir(directory):
		if labels == 'glacier':
			label=2
		elif labels == 'sea':
			label=4
		elif labels == 'buidings':
			label=0
		elif labels == 'forest':
			label=1
		elif labels == 'street':
			label=5
		elif labels == 'mountain':
			label=3
	
	for i in os.listdir(directory+labels):
		image=cv2.imread(directory+labels+r'/'+i)
		image=cv2.resize(image,(150,150))
		Images.append(image)
		Labels.append(label)
	return shuffle(Images,Labels,random_state=817328462)

def get_classlabel(class_code):
		labels={2:'glacier',4:'sea',0:'buildings',1:'forest',5:'street',3:'mountain'}
		return  labels[class_code]

#def model_predict(img, model):
#    img = img.resize((150, 150))

#    x = np.array(img)

#    preds = model.predict(x)
#    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
	u_file=request.files.getlist('file')
	filename=[]
	for file in u_file:
		if file and allowed_file(file.filename):
			filename=secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
			filename.append(filename)
	if request.method == 'POST':
		img = base64_to_pil(request.json)

		preds = model_predict(img, model)

		pred_proba = "{:.3f}".format(np.amax(preds))    
		pred_class = decode_predictions(preds, top=1)   

		result = str(pred_class[0][0][1])               
		result = result.replace('_', ' ').capitalize()
        
		return jsonify(result=result, probability=pred_proba)

	return None

@app.route('/predict/<filename>')
def uploaded_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
