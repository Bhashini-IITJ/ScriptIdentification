{
    "paths": {
        "base_dir": ".",
        "train_dir": "train_data",
        "test_dir": "test_data",
        "save_model_name": "models/efficientnet/efficientnetv2_m_sliced.pth"
    },
    "hyperparameters": {
        "batch_size": 32,
        "epochs": 30,
        "learning_rate": 0.0001,
        "image_size": 480,
        "seed": 42,
        "val_split": 0.2
    },
    "classes": [
        "hindi", "english", "assamese", "bengali",
        "gujarati", "kannada", "malayalam", "marathi",
        "odia", "punjabi", "tamil", "telugu"
    ]
}
