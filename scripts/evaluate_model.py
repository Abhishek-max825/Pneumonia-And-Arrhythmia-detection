import torch
from train_xray import build_model
from utils.preprocessing import get_xray_loaders
from tqdm import tqdm

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Data
    _, _, test_loader, _ = get_xray_loaders('data/xray', batch_size=32)
    
    # Load Model
    model = build_model(device)
    try:
        model.load_state_dict(torch.load('models/xray_model.pt', map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: models/xray_model.pt not found.")
        return

    model.eval()
    
    running_corrects = 0
    total = 0
    
    print("Evaluating on Test Set...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            
            running_corrects += torch.sum(preds == labels.data)
            total += inputs.size(0)
            
    acc = running_corrects.double() / total
    print(f"Test Accuracy: {acc:.4f} ({running_corrects}/{total})")

if __name__ == "__main__":
    evaluate()
