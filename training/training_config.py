from pydantic import BaseModel


class TrainingConfig(BaseModel):
    model_checkpoint_path: str = "./model.pth"
    data_root_dir: str = "./data"
    batch_size: int = 4096
    learning_rate: float = 0.004
    weight_decay: float = 0.05
    num_epochs: int = 300
    patience: int = 10
