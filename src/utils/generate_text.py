# Text generation function
import torch
from torch.nn.functional import softmax

# def generate_text(model, seed_text, char2idx, idx2char, seq_length, num_chars=1000, temperature=0.8, device='cuda'):
#     model.eval()
#     current_sequence = [char2idx[char] for char in seed_text]
#     generated_text = seed_text
    
#     with torch.no_grad():
#         for _ in range(num_chars):
#             # Prepare input
#             if len(current_sequence) > seq_length:
#                 current_sequence = current_sequence[-seq_length:]
            
#             x = torch.LongTensor([current_sequence]).to(device)
#             hidden = model.init_hidden(1, device)
            
#             # Get prediction
#             output, hidden = model(x, hidden)
#             output = output[:, -1, :] / temperature
            
#             # Sample from the output distribution
#             probs = torch.softmax(output, dim=-1)
#             next_char_idx = torch.multinomial(probs, 1).item()
            
#             # Add to generated text
#             generated_text += idx2char[next_char_idx]
#             current_sequence.append(next_char_idx)
    
#     return generated_text

def generate_text(model, seed_text, char2idx, idx2char, seq_length, num_chars=1000, 
                 temperature=0.8, device='cuda', top_k=None):
    """
    Generate text using the trained model with various sampling strategies
    """
    model.eval()
    
    # Convert seed text to indices
    current_sequence = [char2idx.get(char, char2idx['<UNK>']) for char in seed_text]
    generated_text = seed_text
    
    # Make sure sequence is right length
    if len(current_sequence) > seq_length:
        current_sequence = current_sequence[-seq_length:]
    elif len(current_sequence) < seq_length:
        # Pad with UNK if necessary
        current_sequence = [char2idx['<UNK>']] * (seq_length - len(current_sequence)) + current_sequence
    
    with torch.no_grad():
        for _ in range(num_chars):
            # Prepare input
            x = torch.LongTensor([current_sequence]).to(device)
            hidden = model.init_hidden(1, device)
            
            # Get prediction
            output, hidden = model(x, hidden)
            output = output[:, -1, :] / temperature
            
            # Apply top-k sampling if specified
            if top_k is not None:
                # Zero out all logits below the top k values
                top_k_logits, top_k_indices = torch.topk(output, top_k)
                mask = torch.zeros_like(output).scatter_(1, top_k_indices, 1)
                output = output * mask + -1e10 * (1 - mask)
            
            # Get probability distribution
            probs = softmax(output, dim=-1)
            
            # Sample from the distribution
            next_char_idx = torch.multinomial(probs, 1).item()
            
            # Add to generated text
            next_char = idx2char.get(next_char_idx, '<UNK>')
            generated_text += next_char
            
            # Update sequence
            current_sequence = current_sequence[1:] + [next_char_idx]
    
    return generated_text