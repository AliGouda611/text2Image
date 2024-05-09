from flask import Flask, request, send_file, jsonify
from io import BytesIO
from diffusers import DiffusionPipeline
import torch

# Create a Flask app
app = Flask(__name__)

# Initialize the Stable Diffusion pipeline
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

# Define the HTTP endpoint for generating images
@app.route("/generate-image", methods=["POST"])
def generate_image():
    # Get the prompt from the request
    data = request.json
    prompt = data.get("prompt", "")

    # Generate the image using the prompt
    image = pipe(prompt=prompt).images[0]

    # Save the image to a BytesIO object and send it back as a response
    img_io = BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)

    # Return the image as a PNG file
    return send_file(img_io, mimetype="image/png")

# Define a simple health check endpoint
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(status="ok"), 200

# Run the Flask server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
