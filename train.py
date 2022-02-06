import json
import numpy as np
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import ChatDataset
from models.mlp import MLPNet
from utils.extract_data import preprocess_data
from utils.nltk_utils import bag_of_words
from utils.trainer import train, test

if __name__ == "__main__":
    with open('dataset/intents.json', 'r') as f:
        intents = json.load(f)
    
    with open('dataset/validation.json', 'r') as f:
        validation = json.load(f)
    
    dictionaries, labels, data = preprocess_data(intents)
    _, val_labels, val_data = preprocess_data(validation)
    
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for (pattern_sentence, tag) in data:
        bag = bag_of_words(pattern_sentence, dictionaries)
        X_train.append(bag)
        y_train.append(labels.index(tag))
        
    for (pattern_sentence, tag) in val_data:
        bag = bag_of_words(pattern_sentence, dictionaries)
        X_val.append(bag)
        y_val.append(val_labels.index(tag))
    
    X_train = np.array(X_train)
    y_train =  np.array(y_train)
    X_val =  np.array(X_val)
    y_val =  np.array(y_val)
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).long()
    
    batch_size = 10
    hidden_size = 32
    input_size = len(X_train[0])
    output_size = len(labels)
    learning_rate = 0.01
    epochs = 50
    
    dataset = ChatDataset(X_train, y_train)
    val_dataset = ChatDataset(X_val, y_val)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPNet(input_size, hidden_size, output_size).to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch}')      
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = test(model, val_loader, criterion, device)
    
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": dictionaries,
        "tags": labels
    }
    
    FILE = "data.pth"
    torch.save(data, FILE)
    print(f"Training complete, file saved to {FILE}")