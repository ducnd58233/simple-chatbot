import torch
import torch.nn as nn
import torch.utils.data as td
torch.manual_seed(123)

def train(model, data_loader, optimizer, loss_criteria, device):
    model.train()
    train_loss = 0
    
    for batch, (data, target) in enumerate(data_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        
        out = model(data).to(device)
        loss = loss_criteria(out, target).to(device)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
    avg_loss = train_loss / (batch + 1)
    print(f'Training set: Average loss: {avg_loss:.6f}')
    return avg_loss

def test(model, data_loader, loss_criteria, device):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        batch_count = 0
        
        for batch, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target.to(device)
            batch_count += 1
            out = model(data).to(device)
            
            test_loss += loss_criteria(out, target).item()
            
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target == predicted).item()
    length = len(data_loader.dataset)
    avg_loss = test_loss/batch_count
    acc = 100. * correct / len(data_loader.dataset)
    print(f"Validation set: Average loss: {avg_loss:.6f}, Accuracy: {correct}/{length} ({acc:.2f}%)\n")
    return avg_loss