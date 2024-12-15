from utils.train_model import train_model ,parse_arguments

def main():
    """
    Main entry point for the script
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Train models
    train_model(args)

if __name__ == '__main__':
    main()