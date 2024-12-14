import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

###################
# Utility Functions
###################

def visualize_augmentations(dataset, samples=36):
    import matplotlib.pyplot as plt
    import random
    
    plt.figure(figsize=(10, 10))
    for i in range(samples):
        idx = random.randint(0, len(dataset)-1)
        data = dataset[idx][0]
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if data.shape[0] == 1:  # If channels first, move to last
            data = np.transpose(data, (1, 2, 0))
        plt.subplot(6, 6, i + 1)
        plt.imshow(data.squeeze(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def train(model, device, train_loader, optimizer, criterion, epoch, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Step the scheduler
        if scheduler is not None:
            scheduler.step()

        pred = pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc=f'Epoch={epoch} Loss={loss.item():.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    return train_loss/len(train_loader), 100*correct/processed

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    
    return test_loss, accuracy


