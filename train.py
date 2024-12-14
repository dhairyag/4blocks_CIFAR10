from tqdm import tqdm
import torch
import torch.nn.functional as F

###################
# Training Functions
###################

def train(model, device, train_loader, optimizer, epoch, scheduler=None):
    model.train()
    pbar = tqdm(train_loader)
    total = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if scheduler:
            scheduler.step()
            
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        total += len(target)
        correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / total
        
        pbar.set_description(desc= f'Training set: loss={loss.item():.5f} accuracy={accuracy:.2f}% batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    max_correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if correct > max_correct:
                max_correct = correct

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.5f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return max_correct*100/len(test_loader.dataset)
