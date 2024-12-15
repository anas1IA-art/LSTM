
# import numpy as np
from torch.nn.functional import softmax
import matplotlib.pyplot as plt


def plot_metrics(metrics_dict, save_path=None):
    """
    Plot training metrics over time
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics')
    
    # Plot loss
    axs[0, 0].plot(metrics_dict['train_loss'], label='Train')
    axs[0, 0].plot(metrics_dict['val_loss'], label='Validation')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    
    # Plot perplexity
    axs[0, 1].plot(metrics_dict['train_perplexity'], label='Train')
    axs[0, 1].plot(metrics_dict['val_perplexity'], label='Validation')
    axs[0, 1].set_title('Perplexity')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Perplexity')
    axs[0, 1].legend()
    
    # Plot accuracy
    axs[1, 0].plot(metrics_dict['train_accuracy'], label='Train')
    axs[1, 0].plot(metrics_dict['val_accuracy'], label='Validation')
    axs[1, 0].set_title('Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()