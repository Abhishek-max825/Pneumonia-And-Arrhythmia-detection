import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import os
from train_xray import build_model

def load_xray_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    model.load_state_dict(torch.load('models/xray_model.pt', map_location=device))
    model.eval()
    return model, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def test_image(image_path):
    print(f"Testing: {image_path}")
    if not os.path.exists(image_path):
        print("File not found.")
        return

    image = Image.open(image_path).convert('RGB')
    model, device = load_xray_model()
    tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        # Original
        output = model(tensor)
        prob = torch.sigmoid(output).item()
        
        # Flip
        tensor_flip = preprocess_image(image.transpose(Image.FLIP_LEFT_RIGHT)).to(device)
        output_flip = model(tensor_flip)
        prob_flip = torch.sigmoid(output_flip).item()
    
    print(f"Raw Probability: {prob:.4f}")
    print(f"Flip Probability: {prob_flip:.4f}")
    avg_prob = (prob + prob_flip) / 2
    print(f"Avg Probability: {avg_prob:.4f}")
    
    final_prob = avg_prob

    if final_prob > 0.5:
        print(f"Prediction: PNEUMONIA ({final_prob:.2%})")
    else:
        print(f"Prediction: NORMAL ({(1-prob):.2%})")

# Run on the uploaded image if possible, or a placeholder
# I need to know the path of the uploaded image. 
# The user's uploaded image is likely in a temp dir or I can try to find it.
# But for now, I'll just check if I can run this logic.
# Wait, I can't access the user's browser-uploaded file from here easily unless I saved it.
# But I did save it to 'st.session_state'. It's in memory.
# However, I can't run this script on the User's memory.

# But wait! I checked artifacts.
# The artifacts listing shows:
# [ARTIFACT: uploaded_image_1768573024525]
# Path: file:///C:/Users/abhis/.gemini/antigravity/brain/c33294e2-dc64-46f8-8d9c-9f5e7b134243/uploaded_image_1768573024525.png
# This IS the image! I can test it!

test_image(r"C:/Users/abhis/.gemini/antigravity/brain/c33294e2-dc64-46f8-8d9c-9f5e7b134243/uploaded_image_1768573024525.png")
