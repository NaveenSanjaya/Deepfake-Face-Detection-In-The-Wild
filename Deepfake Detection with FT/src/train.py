import os
import torch   
from tqdm.auto import tqdm

from torch.amp import GradScaler
from torchmetrics import Accuracy
from pathlib import Path

scaler = GradScaler()


def train(model, dataloader, criterion, optimizer, num_epochs, checkpoint_dir):
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    #accuracy = Accuracy(task='binary').to(device)
    correct = 0
    total = 0

    for epoch in tqdm(range(num_epochs)):
        training_loss = 0.0
        #accuracy.reset()
        
        for batch_idx, (inputs, labels) in enumerate(tqdm(dataloader)):
                labels = labels.float().to(device)
                inputs = inputs.to(device)

                outputs = model(inputs)
                loss = criterion(outputs.view(-1), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

                 # Update accuracy metric
                predicted = (outputs.view(-1) > 0.5).float()
                #accuracy.update(predicted, labels)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if batch_idx % 50 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
                torch.cuda.empty_cache()  # Clear CUDA cache to prevent memory leaks

        epoch_loss = training_loss / len(dataloader)
        # epoch_accuracy = accuracy.compute().item() * 100
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:  # Save every 5 epochs
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at {checkpoint_path}')
    
    # Save final model
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "model_01_resnet50.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
                f=MODEL_SAVE_PATH)


    # final_checkpoint_path = os.path.join(checkpoint_dir, 'final_checkpoint.pth')
    # torch.save(model.state_dict(), final_checkpoint_path)
    # print(f'Final checkpoint saved at {final_checkpoint_path}')

def main():
    pass

if __name__ == "__main__":
    main()