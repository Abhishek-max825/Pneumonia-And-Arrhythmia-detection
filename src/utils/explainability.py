import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Ensure input requires gradient so the graph is built
        if not x.requires_grad:
            x.requires_grad = True
            
        self.model.eval()
        output = self.model(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        
        output.backward(gradient=one_hot, retain_graph=True)
        
        if self.gradients is None:
            print("❌ Error: Gradients not captured! Check if hooks are registered and graph is connected.")
            # Return empty CAM
            input_spatial = x.shape[2:] if len(x.shape) == 4 else (x.shape[2], 1) # Fallback dimensions
            return np.zeros(input_spatial, dtype=np.float32), 0.0

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Calculate weights based on feature map dimensions
        # 2D: (C, H, W) -> mean over (1, 2)
        # 1D: (C, L) -> mean over (1,)
        axis = tuple(range(1, len(gradients.shape)))
        weights = np.mean(gradients, axis=axis)
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        
        # Resize to input size
        if len(x.shape) == 4: # 2D Image (B, C, H, W)
            input_h, input_w = x.shape[2], x.shape[3]
            cam = cv2.resize(cam, (input_w, input_h))
        elif len(x.shape) == 3: # 1D Signal (B, C, L)
            input_l = x.shape[2]
            cam = cv2.resize(cam.reshape(1, -1), (input_l, 1)).flatten()
        
        # Normalize
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, output.sigmoid().item() if output.shape[1] == 1 else output.softmax(dim=1)[0][class_idx].item()

def overlay_cam(img_tensor, cam, alpha=0.5):
    """
    Overlays CAM on image tensor.
    img_tensor: (3, H, W) normalized tensor
    cam: (H, W) float array in [0, 1]
    """
    # Denormalize Image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = img_tensor.detach().permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1] # BGR to RGB
    
    overlayed = heatmap * alpha + img * (1 - alpha)
    overlayed = np.clip(overlayed, 0, 1)
    
    return overlayed
