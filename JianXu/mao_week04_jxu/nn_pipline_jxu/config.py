from pathlib import Path

"""
配置参数
"""

BASE_DIR = Path(__file__).resolve().parent # 当前文件（config.py）的路径,转成绝对路径,到所在目录

Config = {
    "model_path": (BASE_DIR / "output").resolve(),
    "train_data_path": (BASE_DIR / "data/train_tag_news.json").resolve(),
    "valid_data_path": (BASE_DIR / "data/valid_tag_news.json").resolve(),
    "vocab_path": (BASE_DIR / "chars.txt").resolve(),
    "model_type": "bert",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path": (BASE_DIR / "../../week06/bert-base-chinese").resolve(),
    "seed": 987,
}
