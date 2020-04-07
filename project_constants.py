from libs.experiments import state_dict_hyperparameters, state_dict_hyperparameters_multi

# File paths
DATASET_INFO_PATH = "data/dataset_info.pt"
DATASET_TAG_INFO_PATH = "data/dataset_tag_info.pt"
DATASET_FILE_PATHS = ["data/allabcwrepeats_parsed.txt"]

# Split percentages for dataset partitions
DATASET_SPLIT = (0.88, 0.1, 0.02)

# Choose which experimental models to test with
_MODEL_TO_USE = 2
_MODEL_TO_USE_MULTI = 3


# Choose parameters based on model
STATE_DICT_PATH = state_dict_hyperparameters[_MODEL_TO_USE]["STATE_DICT_PATH"]
MODEL_HIDDEN_SIZE = state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_HIDDEN_SIZE"]
MODEL_EMBEDDING_SIZE = state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_EMBEDDING_SIZE"]
MODEL_EMBEDDINGS_DROPOUT = state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_EMBEDDINGS_DROPOUT"]
MODEL_LSTM_DROPOUT = state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_LSTM_DROPOUT"]
MODEL_NUM_HIDDEN_LAYERS = state_dict_hyperparameters[_MODEL_TO_USE]["MODEL_NUM_HIDDEN_LAYERS"]

# Same for multi-task
STATE_DICT_PATH_MULTI = state_dict_hyperparameters_multi[_MODEL_TO_USE_MULTI]["STATE_DICT_PATH"]
MODEL_HIDDEN_SIZE_MULTI = state_dict_hyperparameters_multi[_MODEL_TO_USE_MULTI]["MODEL_HIDDEN_SIZE"]
MODEL_EMBEDDING_SIZE_MULTI = state_dict_hyperparameters_multi[_MODEL_TO_USE_MULTI]["MODEL_EMBEDDING_SIZE"]
MODEL_EMBEDDINGS_DROPOUT_MULTI = state_dict_hyperparameters_multi[_MODEL_TO_USE_MULTI]["MODEL_EMBEDDINGS_DROPOUT"]
MODEL_LSTM_DROPOUT_MULTI = state_dict_hyperparameters_multi[_MODEL_TO_USE_MULTI]["MODEL_LSTM_DROPOUT"]
MODEL_NUM_HIDDEN_LAYERS_MULTI = state_dict_hyperparameters_multi[_MODEL_TO_USE_MULTI]["MODEL_NUM_HIDDEN_LAYERS"]
STRUCTURE_TASK_WEIGHT = state_dict_hyperparameters_multi[_MODEL_TO_USE_MULTI]["STRUCTURE_TASK_WEIGHT"]
