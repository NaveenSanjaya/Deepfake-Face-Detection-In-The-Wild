import torch
from models import Model
from datasets import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def load_model(model, checkpoint_path):
    
    model.load_state_dict(torch.load(checkpoint_path))

    model.eval()  # Set the model to evaluation mode
    return model

def evaluate(model, valid_loader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    with torch.inference_mode():

        for images, labels in valid_loader:
            labels = labels.float().to(device)
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total

    conf_matrix = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = conf_matrix.ravel()

    print(f'Accuracy of the model on the validation images: {accuracy:.2f}%')
    print(f'True Negatives (Real identified as Real): {tn}')
    print(f'False Positives (Real identified as Fake): {fp}')
    print(f'False Negatives (Fake identified as Real): {fn}')
    print(f'True Positives (Fake identified as Fake): {tp}')
    print('Confusion Matrix:')
    print(conf_matrix)

def main():
    pass
    """# Define transformations for the test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.3996, 0.3194, 0.3223], [0.2321, 0.1766, 0.1816])
    ])

    # Load the test dataset
    valid_dataset = Dataset(transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Load the trained model
    model = load_model('path/to/the/checkpoint.pth')
    
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    # Evaluate the model
    evaluate(model, valid_loader)"""

if __name__ == '__main__':
    main()