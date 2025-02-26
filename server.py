from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)

# Load the reference image (expected eye image)
reference_image_path = "expected_eye_image.jpeg"
reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)

def decode_image(image_data):
    """Convert base64 image data to OpenCV format"""
    image_data = image_data.split(",")[1]  # Remove data:image/png;base64,
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

def compare_images(image1, image2):
    """Compare two images using Structural Similarity Index (SSIM)"""
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))  # Resize to match dimensions
    score, _ = ssim(image1, image2, full=True)
    return score

@app.route("/compare", methods=["POST"])
def compare():
    data = request.json
    captured_image = decode_image(data["image"])

    similarity_score = compare_images(captured_image, reference_image)

    return jsonify({"match": similarity_score > 0.75})  # Threshold for a match

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
