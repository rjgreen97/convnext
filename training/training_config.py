from pydantic import BaseModel


class TrainingConfig(BaseModel):
    model_checkpoint_path: str = "./model.pth"
    data_root_dir: str = "./data"
    batch_size: int = 512
    learning_rate: float = 0.0005
    weight_decay: float = 0.0001
    num_epochs: int = 300
    patience: int = 10
