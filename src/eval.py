import torch

# def evaluate_model(model, val_loader, criterion, device):
#     model.eval()
#     total_loss = 0
    
#     with torch.no_grad():
#         for inputs, targets in val_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             batch_size = inputs.size(0)
            
#             hidden = model.init_hidden(batch_size, device)
#             output, hidden = model(inputs, hidden)
            
#             output = output.view(-1, model.vocab_size)
#             targets = targets.view(-1)
            
#             loss = criterion(output, targets)
#             total_loss += loss.item()
            
#     return total_loss / len(val_loader)

import torch
# import numpy as np
# from torch.nn.functional import softmax
# import matplotlib.pyplot as plt
# from datetime import datetime

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate model performance with multiple metrics
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    perplexity = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            
            # Initialize hidden state
            hidden = model.init_hidden(batch_size, device)
            
            # Forward pass
            output, _ = model(inputs, hidden)
            
            # Reshape output and targets
            output_flat = output.view(-1, model.vocab_size)
            targets_flat = targets.view(-1)
            
            # Calculate loss
            loss = criterion(output_flat, targets_flat)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = output_flat.argmax(dim=1)
            correct_predictions += (predictions == targets_flat).sum().item()
            total_predictions += targets_flat.size(0)
            
            # Calculate perplexity
            perplexity += torch.exp(loss).item()
    
    # Calculate final metrics
    avg_loss = total_loss / len(test_loader)
    accuracy = correct_predictions / total_predictions
    avg_perplexity = perplexity / len(test_loader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'perplexity': avg_perplexity
    }