import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request


app = Flask(__name__)

model = load_model('D:/New folder (2)/OneDrive/desktop/test/ECG.h5')

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    text = ""
    if request.method == 'POST':
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)
        img = image.load_img(filepath, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = np.argmax(model.predict(x), axis=1)
        if pred == 0:
            text = "Left Bundle Branch block"
        elif pred == 1:
            text = "Normal"
        elif pred == 2:
            text = "Premature Atrial Contraction"
        elif pred == 3:
            text = "Premature Ventricular Contraction"
        elif pred == 4:
            text = "Right Bundle Branch Block"
        else:
            text = "Ventricular Fibrillation"
    return text

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == '__main__':
    app.run(debug=False, port=8080)  # Change the port to 8080
