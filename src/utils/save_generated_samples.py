from datetime import datetime

from src.utils import generate_text


def save_generated_samples(model_name, model, dataset, device, sample_length=1000,
                         temperatures=[0.5, 0.7, 1.0, 1.2], top_k_values=[None, 5, 10]):
    """
    Generate and save multiple text samples with different parameters
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'samples_{model_name}_{timestamp}.txt'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Generated Samples from {model_name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Generate samples with different temperatures and top-k values
        for temp in temperatures:
            for top_k in top_k_values:
                f.write(f"\nTemperature: {temp}, Top-k: {top_k}\n")
                f.write("-" * 30 + "\n")
                
                sample = generate_text(
                    model=model,
                    seed_text="The",
                    char2idx=dataset.char2idx,
                    idx2char=dataset.idx2char,
                    seq_length=dataset.seq_length,
                    num_chars=sample_length,
                    temperature=temp,
                    device=device,
                    top_k=top_k
                )
                
                f.write(sample + "\n")
                f.write("-" * 50 + "\n")
    
    return filename