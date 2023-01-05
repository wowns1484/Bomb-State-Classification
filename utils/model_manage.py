import os
import torch

def save_model(model, saved_dir, file_name):
    os.makedirs(saved_dir, exist_ok=True)
    
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)
    print("Saved model.")
    
def load_model(args, model):
    checkpoint = torch.load(args.model_path, map_location=args.device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    print("Loaded model.")
    
    return model