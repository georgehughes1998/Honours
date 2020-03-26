# Some experimental models with different hyperparameters
state_dict_hyperparameters = {

    # Calculated perplexity: 5.440979859545349
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

    # Calculated perplexity: 5.414186633376884
    4: {"STATE_DICT_PATH": "model/state_dict_04.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.6,
        "MODEL_LSTM_DROPOUT": 0.7,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 4.5686740477528796
    5: {"STATE_DICT_PATH": "model/state_dict_05.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 4.745441007895771
    6: {"STATE_DICT_PATH": "model/state_dict_06.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 2
        },

    # Calculated perplexity: 4.415783155239483
    7: {"STATE_DICT_PATH": "model/state_dict_07.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 4.671028831548884
    8: {"STATE_DICT_PATH": "model/state_dict_08.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 2
        },
}


state_dict_hyperparameters_multi = {

    # Calculated perplexity:
    0: {"STATE_DICT_PATH": "model/state_dict_multi_00.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 1
        },
}