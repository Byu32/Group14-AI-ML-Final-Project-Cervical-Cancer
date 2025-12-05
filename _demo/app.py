import os
import torch
import timm
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as T
import io
import sys
import numpy as np
import base64
import matplotlib.cm as cm
import threading

# Initialize Flask App
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIGURATION ---
CLASSES = {0: "Normal", 1: "Precancer", 2: "Cancer"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PREDICTION_LOCK = threading.Lock()

# Global model variable starts as None (No default model)
model = None

print(f"--- SERVER STARTUP ---")
print(f"Server running. Waiting for Admin to upload a model via the frontend...")

# --- MODEL LOADING HELPER ---
def load_model_from_path(path):
    global model
    print(f"Loading model from {path}...")
    try:
        loaded_model = torch.load(path, map_location=DEVICE)
        loaded_model.eval()
        model = loaded_model
        print("[SUCCESS] Model loaded successfully!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to load model: {e}")
        return False

# --- GRAD-CAM HELPER CLASS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __enter__(self):
        # Register hooks
        self.handles.append(self.target_layer.register_forward_hook(self.save_activation))
        self.handles.append(self.target_layer.register_full_backward_hook(self.save_gradient))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Remove hooks
        for handle in self.handles:
            handle.remove()

def generate_heatmap_overlay(grads, acts, original_image):
    # Pool the gradients across the channels
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    # Weight the channels by corresponding gradients
    acts = acts[0] # remove batch dim
    for i in range(acts.shape[0]):
        acts[i, :, :] *= pooled_grads[i]
    
    # Average the channels of the activations
    heatmap = torch.mean(acts, dim=0).cpu().detach().numpy()
    
    # ReLU on top of the heatmap
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize the heatmap
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    
    # Resize heatmap to match original image size
    img_w, img_h = original_image.size
    # We use PIL to resize the heatmap to image dimensions
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap))
    heatmap_img = heatmap_img.resize((img_w, img_h), Image.BICUBIC)
    
    # Apply colormap (Jet is common for heatmaps)
    heatmap_array = np.array(heatmap_img) / 255.0
    colormap = cm.get_cmap("jet")
    colored_heatmap = colormap(heatmap_array) # Returns RGBA
    
    # Convert to PIL Image (discard alpha from colormap for blending if needed, but we keep it)
    colored_heatmap = Image.fromarray((colored_heatmap[:, :, :3] * 255).astype(np.uint8))
    
    # Blend images
    overlayed_img = Image.blend(original_image.convert("RGB"), colored_heatmap, alpha=0.4)
    return overlayed_img

# --- TRANSFORMS ---
def get_transform():
    image_size = 384
    return T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

transform = get_transform()

# --- ROUTES ---

@app.route('/predict', methods=['POST'])
def predict():
    global model
    print("Received a prediction request...")

    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'No model is currently loaded. Please upload one via Admin.'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # 1. Open Image
        img_bytes = file.read()
        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # 2. Transform Image
        tensor = transform(original_img).unsqueeze(0).to(DEVICE)
        tensor.requires_grad = True # Enable gradients for input
        
        # 3. Identify Target Layer
        target_layer = getattr(model, 'conv_head', None) 
        if target_layer is None:
            target_layer = list(model.modules())[-2]

        gradcam_b64 = None

        # 4. Run Inference with Grad-CAM hooks
        with PREDICTION_LOCK: # Lock to ensure hooks don't conflict between requests
            with GradCAM(model, target_layer) as cam:
                # Forward Pass
                model.zero_grad()
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Get Top Prediction
                confidence, predicted_class = torch.max(probs, 1)
                class_idx = predicted_class.item()
                conf_score = confidence.item() * 100
                prediction_name = CLASSES.get(class_idx, "Unknown")
                
                # Backward Pass for Grad-CAM
                score = outputs[0, class_idx]
                score.backward()
                
                # Generate Heatmap
                if cam.gradients is not None and cam.activations is not None:
                    overlay_img = generate_heatmap_overlay(cam.gradients, cam.activations, original_img)
                    
                    # Convert to Base64
                    buffered = io.BytesIO()
                    overlay_img.save(buffered, format="PNG")
                    gradcam_b64 = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

        print(f"Predicted: {prediction_name} ({conf_score:.2f}%)")
        
        # 5. Return JSON
        return jsonify({
            'prediction': prediction_name,
            'confidence': f"{conf_score:.2f}%",
            'class_id': class_idx,
            'gradcam_image': gradcam_b64 # Send the image back!
        })

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = file.filename
    
    if not (filename.endswith('.pkl') or filename.endswith('.pt') or filename.endswith('.pth')):
         return jsonify({'error': 'Invalid file type. Please upload .pkl, .pt, or .pth'}), 400

    try:
        # Save file locally
        save_path = "uploaded_custom_model.pkl"
        file.save(save_path)
        
        # Reload model
        success = load_model_from_path(save_path)
        
        if success:
            return jsonify({'message': 'Model updated successfully', 'filename': filename})
        else:
            return jsonify({'error': 'File saved but failed to load into PyTorch.'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)