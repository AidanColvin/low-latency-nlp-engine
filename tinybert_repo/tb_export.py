import torch
from transformers import AutoModelForSequenceClassification

def export_to_torchscript(model_name: str, save_path: str) -> None:
    """
    given a huggingface model name and save path string
    return nothing
    converts pytorch model to c++ compatible format
    saves traced model to disk
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    dummy_input = torch.randint(0, 2000, (1, 256))
    
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(save_path)
