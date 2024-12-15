

def load_data(file_path, train_size_mb=96, val_size_mb=4):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Convert MB to bytes
            train_size = train_size_mb * 1024 * 1024
            val_size = val_size_mb * 1024 * 1024
            
            # Read training data
            text = f.read(train_size)
            print(f"Loaded {len(text)} characters for training")
            
            # Read validation data
            val_text = f.read(val_size)
            print(f"Loaded {len(val_text)} characters for validation")
            
            return text, val_text
    except FileNotFoundError:
        print(f"Error: Could not find file {file_path}")
        raise
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise