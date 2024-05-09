from flask import Flask, request, send_file
from io import BytesIO
from diffusers import DiffusionPipeline
import torch

app = Flask(__name__)

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
pipe.to("cuda")

@app.route("/generate-image", methods=["POST"])
def generate_image():
    data = request.json
    prompt = data.get("prompt", "")
    image = pipe(prompt=prompt).images[0]

    # Return the image as a response
    img_io = BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
