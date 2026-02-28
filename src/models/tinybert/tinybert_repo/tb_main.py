import tb_config
from tb_export import export_to_torchscript

def main() -> None:
    """
    given nothing
    return nothing
    executes tinybert export pipeline
    """
    export_to_torchscript(tb_config.MODEL_NAME, tb_config.EXPORT_FILE)

if __name__ == "__main__":
    main()
