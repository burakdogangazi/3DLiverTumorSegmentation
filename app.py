from flask import Flask, request,render_template,redirect,url_for
from Segmenter import transform_nii,get_result
import nibabel as nib
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt

app = Flask(__name__) 

ALLOWED_EXTENSIONS = {'nii','nii.gz'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict', methods=['POST'])
def predict():

    try:
        file = request.files.get('file')
        nii_bytes = file.read()
        
        tensor = transform_nii(nii_bytes)
        result = get_result(tensor)

        data = {
            'result': result, 
        }
        return render_template('result.html', **data)


    except:
        return redirect(url_for('error'))
        

@app.route('/error', methods=['GET'])
def error():      
    return render_template('error.html')

@app.route('/')
def index():
    return render_template('index.html')

