
# Some experimental models with different hyperparameters
_state_dict_hyperparameters = {

    # Calculated perplexity:
    0: {"STATE_DICT_PATH": "model/state_dict_00.pt",
        "MODEL_HIDDEN_SIZE": 64,
        "MODEL_EMBEDDING_SIZE": 64,
        "MODEL_EMBEDDINGS_DROPOUT": 0.3,
        "MODEL_LSTM_DROPOUT": 0.5,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 6.267296195073176
    1: {"STATE_DICT_PATH": "model/state_dict_01.pt",
        "MODEL_HIDDEN_SIZE": 32,
        "MODEL_EMBEDDING_SIZE": 32,
        "MODEL_EMBEDDINGS_DROPOUT": 0.3,
        "MODEL_LSTM_DROPOUT": 0.5,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 4.8019283890479985
    2: {"STATE_DICT_PATH": "model/state_dict_02.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.3,
        "MODEL_LSTM_DROPOUT": 0.5,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 6.159301027090821
    3: {"STATE_DICT_PATH": "model/state_dict_03.pt",
        "MODEL_HIDDEN_SIZE": 64,
        "MODEL_EMBEDDING_SIZE": 64,
        "MODEL_EMBEDDINGS_DROPOUT": 0.3,
        "MODEL_LSTM_DROPOUT": 0.5,
        "MODEL_NUM_HIDDEN_LAYERS": 2
        },

    # Calculated perplexity:
    4: {"STATE_DICT_PATH": "model/state_dict_04.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.6,
        "MODEL_LSTM_DROPOUT": 0.7,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },
}


DATASET_INFO_PATH = "data/dataset_info.pt"
DATASET_FILE_PATHS = ["data/allabcwrepeats_parsed.txt"]


# Choose which experimental model to test with
_MODEL_TO_USE = 4

STATE_DICT_PATH = _state_dict_hyperparameters[_MODEL_TO_USE]["STATE_DICT_PATH"]

MODEL_HIDDEN_SIZE = _state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_HIDDEN_SIZE"]
MODEL_EMBEDDING_SIZE = _state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_EMBEDDING_SIZE"]
MODEL_EMBEDDINGS_DROPOUT = _state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_EMBEDDINGS_DROPOUT"]
MODEL_LSTM_DROPOUT = _state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_LSTM_DROPOUT"]
MODEL_NUM_HIDDEN_LAYERS = _state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_NUM_HIDDEN_LAYERS"]


