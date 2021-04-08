from flask import Flask, request, jsonify, render_template,redirect, url_for, request,json
from werkzeug.utils import secure_filename
import os
import predict


app = Flask(__name__)

@app.route('/')
def home():
    #return render_template('MapGUI.html')
    return render_template('HomePage.html')

@app.route('/predictGalaxy', methods = ['GET','POST'])
def process_image():
    if request.method == 'POST':
        f = request.files['file']
        location = "static/img/upload/"+f.filename
        f.save(os.path.join('static/img/upload', secure_filename(f.filename)))
        data = {'location': location, 'predict': "round disk like shape"}
        return jsonify(data), 200
      # return jsonify({'status':'OK','location':'found'}) ,200

if __name__ == '__main__':
    app.run(port=8080, debug=True)
