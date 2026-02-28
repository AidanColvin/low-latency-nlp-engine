import fasttext

def train_fasttext_model(input_path: str, output_path: str) -> None:
    """
    given an input file path and output file path
    return nothing
    trains supervised classification model
    saves binary model to disk
    """
    model = fasttext.train_supervised(input=input_path)
    model.save_model(output_path)
