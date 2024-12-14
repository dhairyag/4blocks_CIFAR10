import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from albumentation1 import CIFAR10Albumentations, train_transforms, test_transforms
from model import CIFAR10Net
from utils import train, test
import matplotlib.pyplot as plt
from torchinfo import summary

def main():
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Hyperparameters
        batch_size = 512
        epochs = 28
        base_lr = 0.01
        max_lr = 0.4
        pct_start = 0.25  # Spend 25% of time in ramp up
        div_factor = 25.0  # initial_lr = max_lr/div_factor
        final_div_factor = 3e4  # final_lr = initial_lr/final_div_factor
        
        # Data loading
        train_dataset = CIFAR10Albumentations(
            root='./data', 
            train=True, 
            download=True, 
            transform=train_transforms
        )
        
        test_dataset = CIFAR10Albumentations(
            root='./data', 
            train=False, 
            download=True, 
            transform=test_transforms
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        # Model
        model = CIFAR10Net().to(device)
        # Print model summary and parameters using torchinfo
        print(summary(model, input_size=(1, 3, 32, 32), device=device))
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=max_lr/div_factor, momentum=0.9, weight_decay=5e-4)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        
        # Training and testing
        train_losses = []
        test_losses = []
        train_acc = []
        test_acc = []
        
        # Learning rate history
        lr_history = []
        
        for epoch in range(1, epochs + 1):
            print(f'\nEpoch: {epoch}')
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate: {current_lr:.6f}')
            
            train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch, scheduler)
            test_loss, test_accuracy = test(model, device, test_loader, criterion)
            
            # Save metrics
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_acc.append(train_accuracy)
            test_acc.append(test_accuracy)
            lr_history.append(current_lr)
            
            # Print epoch summary
            # print(f'Train Accuracy: {train_accuracy:.2f}% | Test Accuracy: {test_accuracy:.2f}%')
        
        # Plot training curves
        plt.figure(figsize=(15, 5))
        
        # Plot accuracy
        plt.subplot(1, 3, 1)
        plt.plot(train_acc, label='Train Accuracy')
        plt.plot(test_acc, label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 3, 2)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot learning rate
        plt.subplot(1, 3, 3)
        plt.plot(lr_history)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        
        plt.tight_layout()
        #plt.show()
        
        # Save the plot as a PNG file
        plt.savefig('images/training_curves.png')
        plt.close()

    finally:
        # Clean up DataLoader workers
        train_loader._iterator = None
        test_loader._iterator = None
        
        # Close all pyplot windows
        plt.close('all')

if __name__ == '__main__':
    main() 