import os
from tensorflow.keras.applications.resnet50 import ResNet50
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import tensorflow as tf
UPLOAD_FOLDER = "./static/upload/"
ALLOW_EXTENSIONS = set(["jpg","jpeg","png"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER



def allow_file(file_name):
    return "." in file_name and file_name.rsplit(".", 1)[1].lower() in ALLOW_EXTENSIONS
def preprocess_image(file_path):
    data = load_img(file_path, target_size=(224, 224))
    data = img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)
    return data

@app.route("/", methods = ["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]

        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allow_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("uploaded_file", filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route("/upload/<filename>")
def upload(filename):
    return send_from_directory("upload", filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    tf.keras.backend.clear_session()
    model = ResNet50(weights="imagenet")

    data = preprocess_image(app.config["UPLOAD_FOLDER"]+"/"+filename)
    preds = model.predict(data)
    result = decode_predictions(preds, top=3)[0]
    print(decode_predictions(preds, top=3)[0])

    return render_template("result.html",filename = filename,class1=result[0][1],class2= result[1][1],class3= result[2][1] )


if __name__ == "__main__":
    app.run()