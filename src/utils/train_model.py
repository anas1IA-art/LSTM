import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from models import RNNModel, GRUModel, LSTMModel
from data.Dataset import TextDataset
from .load_data import load_data

def parse_arguments():
    """
    Parse command-line arguments for model training configuration.
    
    Returns:
        Parsed arguments with default values and user-specified overrides
    """
    parser = argparse.ArgumentParser(description='Train RNN, LSTM, and GRU models on text data')
    
    # Data path argument
    parser.add_argument('--data_path', type=str, 
                        default='path',
                        help='Path to the input text data file')
    
    # Model hyperparameters
    parser.add_argument('--seq_length', type=int, default=100,
                        help='Sequence length for input data')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of recurrent layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training hyperparameters
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    
    return parser.parse_args()

def train_model(args):
    """
    Train RNN, LSTM, and GRU models with configurable parameters
    
    Args:
        args (argparse.Namespace): Parsed command-line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Load data
        train_text, val_text = load_data(args.data_path)
        
        # Create datasets
        train_dataset = TextDataset(train_text, args.seq_length)
        val_dataset = TextDataset(val_text, args.seq_length, 
                                vocab=(train_dataset.char2idx, train_dataset.idx2char))
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Model parameters
        vocab_size = len(train_dataset.char2idx)
        
        # Initialize models
        models = {
            'RNN': RNNModel(vocab_size, args.embedding_dim, args.hidden_dim, 
                            args.n_layers, args.dropout).to(device),
            'LSTM': LSTMModel(vocab_size, args.embedding_dim, args.hidden_dim, 
                              args.n_layers, args.dropout).to(device),
            'GRU': GRUModel(vocab_size, args.embedding_dim, args.hidden_dim, 
                            args.n_layers, args.dropout).to(device)
        }
        
        # Training parameters
        criterion = nn.CrossEntropyLoss()
        
        # Train each model
        for name, model in models.items():
            print(f"\nTraining {name} model...")
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            
            for epoch in range(args.n_epochs):
                # Training
                model.train()
                total_train_loss = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    hidden = model.init_hidden(inputs.size(0), device)
                    output, hidden = model(inputs, hidden)
                    
                    # Calculate loss
                    output = output.view(-1, model.vocab_size)
                    targets = targets.view(-1)
                    loss = criterion(output, targets)
                    
                    # Backward pass
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                    
                    if batch_idx % 100 == 0:
                        print(f'Epoch {epoch+1}/{args.n_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')
                
                # Validation
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        hidden = model.init_hidden(inputs.size(0), device)
                        output, hidden = model(inputs, hidden)
                        output = output.view(-1, model.vocab_size)
                        targets = targets.view(-1)
                        loss = criterion(output, targets)
                        total_val_loss += loss.item()
                
                avg_train_loss = total_train_loss / len(train_loader)
                avg_val_loss = total_val_loss / len(val_loader)
                print(f'Epoch {epoch+1}/{args.n_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}')
                
                # Save model checkpoint
                checkpoint_path = f'checkpoints/{name}_epoch_{epoch+1}.pt'
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                }, checkpoint_path)

    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        raise

