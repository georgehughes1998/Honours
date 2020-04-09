# Some experimental models with different hyperparameters
state_dict_hyperparameters = {

    # Calculated perplexity: 6.605727788132069
    0: {"STATE_DICT_PATH": "model/state_dict_00.pt",
        "MODEL_HIDDEN_SIZE": 64,
        "MODEL_EMBEDDING_SIZE": 64,
        "MODEL_EMBEDDINGS_DROPOUT": 0.3,
        "MODEL_LSTM_DROPOUT": 0.5,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 8.216546510823948
    1: {"STATE_DICT_PATH": "model/state_dict_01.pt",
        "MODEL_HIDDEN_SIZE": 32,
        "MODEL_EMBEDDING_SIZE": 32,
        "MODEL_EMBEDDINGS_DROPOUT": 0.3,
        "MODEL_LSTM_DROPOUT": 0.5,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 5.709209776996144
    2: {"STATE_DICT_PATH": "model/state_dict_02.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.3,
        "MODEL_LSTM_DROPOUT": 0.5,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 6.502006394295804
    3: {"STATE_DICT_PATH": "model/state_dict_03.pt",
        "MODEL_HIDDEN_SIZE": 64,
        "MODEL_EMBEDDING_SIZE": 64,
        "MODEL_EMBEDDINGS_DROPOUT": 0.3,
        "MODEL_LSTM_DROPOUT": 0.5,
        "MODEL_NUM_HIDDEN_LAYERS": 2
        },

    # Calculated perplexity: 6.60386709446079
    4: {"STATE_DICT_PATH": "model/state_dict_04.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.6,
        "MODEL_LSTM_DROPOUT": 0.7,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 5.205355115638612
    5: {"STATE_DICT_PATH": "model/state_dict_05.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 5.202738049256018
    6: {"STATE_DICT_PATH": "model/state_dict_06.pt",
        "MODEL_HIDDEN_SIZE": 128,
        "MODEL_EMBEDDING_SIZE": 128,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 2
        },

    # Calculated perplexity: 4.814087963785626
    7: {"STATE_DICT_PATH": "model/state_dict_07.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 4.852525880097235
    8: {"STATE_DICT_PATH": "model/state_dict_08.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 2
        },

    # Calculated perplexity: 4.417362087269969
    9: {"STATE_DICT_PATH": "model/state_dict_09.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0,
        "MODEL_LSTM_DROPOUT": 0,
        "MODEL_NUM_HIDDEN_LAYERS": 1
        },

    # Calculated perplexity: 4.872944001273216
    10: {"STATE_DICT_PATH": "model/state_dict_10.pt",
         "MODEL_HIDDEN_SIZE": 512,
         "MODEL_EMBEDDING_SIZE": 512,
         "MODEL_EMBEDDINGS_DROPOUT": 0.3,
         "MODEL_LSTM_DROPOUT": 0.5,
         "MODEL_NUM_HIDDEN_LAYERS": 1
        },
}


state_dict_hyperparameters_multi = {

    # Calculated perplexity: 4.574427658103038
    # Calculated perplexity for tags: 1.183124633253071
    0: {"STATE_DICT_PATH": "model/state_dict_multi_00.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 1
        },

    # Calculated perplexity: 8.434306174536122
    # Calculated perplexity for tags: 1.307660920955928
    1: {"STATE_DICT_PATH": "model/state_dict_multi_01.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 5
        },

    # Calculated perplexity: 4.401730298283562
    # Calculated perplexity for tags: 1.2026837358945117
    2: {"STATE_DICT_PATH": "model/state_dict_multi_02.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0.5
        },

    # Calculated perplexity: 4.382221332094143
    # Calculated perplexity for tags: 1.3355697331341594
    3: {"STATE_DICT_PATH": "model/state_dict_multi_03.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0.25
        },

    # Calculated perplexity: 4.359360219939451
    # Calculated perplexity for tags: 1.5815429344936534
    4: {"STATE_DICT_PATH": "model/state_dict_multi_04.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0.1
        },

    # Calculated perplexity: 4.3522911747771875
    # Calculated perplexity for tags: 2.12581175856193
    5: {"STATE_DICT_PATH": "model/state_dict_multi_05.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0.05
        },

    # Calculated perplexity: 4.435203962900802
    # Calculated perplexity for tags: 9.737601021920518
    6: {"STATE_DICT_PATH": "model/state_dict_multi_06.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0
        },

    # Calculated perplexity: 4.312212959622073
    # Calculated perplexity for tags: 2.0742064022972406
    7: {"STATE_DICT_PATH": "model/state_dict_multi_07.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0,
        "MODEL_LSTM_DROPOUT": 0,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0.05
        },

    # Calculated perplexity: 4.253170632171199
    # Calculated perplexity for tags: 2.9898213275397127
    8: {"STATE_DICT_PATH": "model/state_dict_multi_08.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0,
        "MODEL_LSTM_DROPOUT": 0,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0.005
        },

    # Calculated perplexity:
    # Calculated perplexity for tags:
    9: {"STATE_DICT_PATH": "model/state_dict_multi_09.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0.1,
        "MODEL_LSTM_DROPOUT": 0.2,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0.005
        },

    # Calculated perplexity:
    # Calculated perplexity for tags:
    10: {"STATE_DICT_PATH": "model/state_dict_multi_10.pt",
        "MODEL_HIDDEN_SIZE": 256,
        "MODEL_EMBEDDING_SIZE": 256,
        "MODEL_EMBEDDINGS_DROPOUT": 0,
        "MODEL_LSTM_DROPOUT": 0,
        "MODEL_NUM_HIDDEN_LAYERS": 1,
        "STRUCTURE_TASK_WEIGHT": 0.0005
        },

    # Calculated perplexity:
    # Calculated perplexity for tags:
    11: {"STATE_DICT_PATH": "model/state_dict_multi_11.pt",
         "MODEL_HIDDEN_SIZE": 256,
         "MODEL_EMBEDDING_SIZE": 256,
         "MODEL_EMBEDDINGS_DROPOUT": 0.1,
         "MODEL_LSTM_DROPOUT": 0.2,
         "MODEL_NUM_HIDDEN_LAYERS": 1,
         "STRUCTURE_TASK_WEIGHT": 0.0005
         },
}
