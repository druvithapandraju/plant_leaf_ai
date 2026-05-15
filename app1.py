import os
import json
import time
import base64
import numpy as np
import cv2

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# ================= APP INIT =================
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ================= LOAD MODEL =================
model = load_model(
    "hybrid_model.h5",
    compile=False
)

# ================= LOAD LABELS =================
with open("labels.json") as f:
    labels = json.load(f)

# ================= SOLUTIONS =================
def get_solution(disease):

    solutions = {
        "early_blight": "Use fungicide and remove infected leaves.",
        "late_blight": "Apply copper-based fungicide immediately.",
        "leaf_spot": "Avoid overhead watering and use neem oil.",
        "rust": "Apply sulfur spray.",
        "powdery_mildew": "Use baking soda spray or fungicide.",
        "brown_spot": "Use appropriate fungicide and avoid excess moisture.",
        "leaf_blight": "Remove infected parts and apply fungicide.",
        "leaf_curl": "Use insecticides and maintain plant hygiene.",
        "anthracnose": "Remove infected leaves and apply fungicide.",
        "black_rot": "Use copper fungicide and prune affected leaves.",
        "scab": "Apply sulfur-based fungicide.",
        "healthy": "Your plant is healthy 🌱"
    }

    return solutions.get(disease, "Maintain proper plant care.")

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/graph")
def graph():
    return render_template("graph.html")

@app.route("/shop")
def shop():
    return render_template("shop.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

# ================= INPUT HANDLERS =================
@app.route("/predict_drag", methods=["POST"])
def predict_drag():
    return handle_prediction()

@app.route("/predict_scan", methods=["POST"])
def predict_scan():
    return handle_prediction()

@app.route("/predict_camera", methods=["POST"])
def predict_camera():

    image_data = request.form.get("image_data")

    if not image_data:
        return redirect(url_for("home"))

    try:
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)

        filename = str(int(time.time() * 1000)) + "_camera.png"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        with open(filepath, "wb") as f:
            f.write(image_bytes)

        result = predict_image(filepath)

        return render_template("result.html", **result)

    except Exception as e:
        return f"Error processing camera image: {e}"

# ================= CORE PREDICTION =================
def predict_image(filepath):

    img = cv2.imread(filepath)

    if img is None:
        return {
            "plant_name": "Error",
            "disease_name": "Invalid Image",
            "confidence": 0,
            "description": "Could not process image.",
            "solution": "Try another image.",
            "original_image": "",
            "infected_image": "",
            "infected_spots": []
        }

    original = img.copy()

    # ================= PREPROCESS =================
    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized.astype("float32") / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # ================= PREDICT =================
    preds = model.predict(img_resized, verbose=0)[0]

    class_index = np.argmax(preds)
    confidence = round(float(preds[class_index]) * 100, 2)

    label = labels[class_index]

    # ================= SPLIT LABEL =================
    if "___" in label:
        plant, disease = label.split("___")

        plant_name = plant.replace("_", " ").title()
        disease_name = disease.replace("_", " ").title()

    else:
        plant_name = "Unknown"
        disease_name = label

    disease_key = disease_name.lower().replace(" ", "_")

    # ================= INFECTED AREA DETECTION =================
    infected_img = original.copy()

    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    infected_spots = []

    if "healthy" not in disease_key:

        lower = np.array([10, 50, 50])
        upper = np.array([35, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:

            if cv2.contourArea(cnt) > 500:

                (x, y), radius = cv2.minEnclosingCircle(cnt)

                x = int(x)
                y = int(y)
                r = int(radius)

                cv2.circle(
                    infected_img,
                    (x, y),
                    r,
                    (0, 0, 255),
                    2
                )

                infected_spots.append({
                    "x": x,
                    "y": y,
                    "r": r
                })

    # ================= SAVE INFECTED IMAGE =================
    base, ext = os.path.splitext(filepath)

    infected_path = base + "_infected" + ext

    cv2.imwrite(infected_path, infected_img)

    return {
        "plant_name": plant_name,
        "disease_name": disease_name,
        "confidence": confidence,
        "description": f"{plant_name} plant is affected by {disease_name}.",
        "solution": get_solution(disease_key),
        "original_image": "/" + filepath.replace("\\", "/"),
        "infected_image": "/" + infected_path.replace("\\", "/"),
        "infected_spots": infected_spots
    }

# ================= FILE HANDLER =================
def handle_prediction():

    if "file" not in request.files:
        return redirect(url_for("home"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("home"))

    filename = (
        str(int(time.time() * 1000))
        + "_"
        + secure_filename(file.filename)
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(filepath)

    result = predict_image(filepath)

    return render_template("result.html", **result)

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
