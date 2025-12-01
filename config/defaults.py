from yacs.config import CfgNode


# Base Config Node
_C = CfgNode()


# === Model Config ===
#   Contains information about selecting and building the model.
_C.MODEL = CfgNode()

# The model architecture to use [NaiveNet, GatedRecurrentUnit]
_C.MODEL.ARCHITECTURE = "NaiveNet"

# The checkpoint to pre-load model weights (.pt) before training. Can also be run or checkpoint dir.
_C.MODEL.CHECKPOINT_PATH = ''


# === Execution Config ===
#   Contains information about running the model.
_C.EXECUTION = CfgNode()

# The execution mode we are running. [train, test, demo]
_C.EXECUTION.MODE = 'train'

# The directory to save the execution run
_C.EXECUTION.OUTPUT_DIR = ''

# The GPU devices to use. Empty list for CPU execution.
_C.EXECUTION.AVAILABLE_GPUS = [0]

# The number of parallel workers to use in the data loader
_C.EXECUTION.NUM_WORKERS = 1

# Mini-Batch Size
_C.EXECUTION.BATCH_SIZE = 16


# === Solver Config ===
#   Contains information about the optimizing the model.
_C.SOLVER = CfgNode()

# Number of Epochs
_C.SOLVER.TOT_EPOCHS = 30

# Period Between Checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 5

# Learning Rate
_C.SOLVER.BASE_LR = 0.01

# Learning rate scheduler (default: constant lr)
_C.SOLVER.LR_SCHEDULER = ''

# Weight decay (default: no decay)
_C.SOLVER.WEIGHT_DECAY = 0.0

# Loss function
_C.SOLVER.LOSS_FN = 'CrossEntropyLoss'

# The Optimizer Method
_C.SOLVER.OPTIMIZER = 'SGD'

# To resume training by picking up at a certain epoch
_C.SOLVER.RESUME = False


# === Dataset / Dataloader Config ===
#   Contains information about the dataset, augmentation, etc
_C.DATA = CfgNode()

# The dataset class to use
_C.DATA.DATASET = 'EmailDatasetCSV'

# The path to the csv containing the data
_C.DATA.CSV_PATH = ''

# The tokenizer to use from torchtext.utils.get_tokenizer
_C.DATA.TT_TOKENIZER = 'basic_english'

# The maximum allowable sequence length
_C.DATA.MAX_SEQ_LEN = 6000

# === Naive Net Config ===
#   Used when NaiveNet is selected for MODEL.ARCHITECTURE
_C.NAIVE_NET = CfgNode()

# Embedding dimension
_C.NAIVE_NET.EMBED_DIM = 96

# Dropout Rate for the model
_C.NAIVE_NET.DROPOUT = 0.0


# === Gated Recurrent Unit (GRU) Config ===
#   Used when GatedRecurrentUnit is selected for MODEL.ARCHITECTURE
_C.GRU = CfgNode()

# Embedding dimension
_C.GRU.EMBED_DIM = 96

# Hidden Dimension
_C.GRU.HIDDEN_DIM = 64

# Number of Layer
_C.GRU.NUM_LAYERS = 2

# Whether to use bidirectional embedding.
_C.GRU.BI_DIR = False

# === Long Short Term Memory (LSTM) Config ===
#   Used when LSTM is selected for MODEL.ARCHITECTURE
_C.LSTM = CfgNode()

# Embedding dimension
_C.LSTM.EMBED_DIM = 96

# Hidden Dimension
_C.LSTM.HIDDEN_DIM = 64

# Number of Layer
_C.LSTM.NUM_LAYERS = 2

# Whether to use bidirectional embedding.
_C.LSTM.BI_DIR = False

# Dropout Rate for the model
_C.LSTM.DROPOUT = 0.1

# === Transformer Config ===
#   Used when Tranformer is selected for MODEL.ARCHITECTURE
_C.TRANSFORMER = CfgNode()

# Embedding dimension
_C.TRANSFORMER.EMBED_DIM = 96

# Hidden Dimension
_C.TRANSFORMER.HIDDEN_DIM = 64

# Number of Layer
_C.TRANSFORMER.NUM_LAYERS = 1

# Dropout Rate for the model
_C.TRANSFORMER.DROPOUT = 0.1

# Number of Self Attention Heads
_C.TRANSFORMER.NHEAD = 1

# === Head Config ===
#   Defines the head module (the last part of the model that does prediction)
_C.HEAD = CfgNode()

# Head Type (linear layer, mlp etc)
_C.HEAD.TYPE = 'linear'

# Number of Predicted Classes
_C.HEAD.NUM_CLASSES = 4

# Defines the MLP layers: [layer_1 output, layer_2 output, etc]
# Last layer output must match NUM_CLASSES
_C.HEAD.LAYERS = []

# Activation function used in between layers
_C.HEAD.LAYER_ACT_FUNC = 'relu'

# Dropout used
_C.HEAD.DROPOUT = 0.0


# ==== LLM Prompted Module Configs ====
_C.PROMPTED_MODULE = CfgNode()

# Prompt file location
_C.PROMPTED_MODULE.PROMPT_FILE = "/home/peter/dev/project_sw/PhishDetect/settings/prompts/phish_detect.txt"

# History len
_C.PROMPTED_MODULE.HISTORY_LEN = 100

# Prompted Module Backend. [LLM_Module, LlamaCpp]
_C.PROMPTED_MODULE.BACKEND = "LLM_Module"

# Sub Node for LLM_Client
_C.PROMPTED_MODULE.LLM_CLIENT = CfgNode()

# Web host to connect with LLM_Server
_C.PROMPTED_MODULE.LLM_CLIENT.HOST = "localhost"

# Web port to connect with LLM_Server
_C.PROMPTED_MODULE.LLM_CLIENT.PORT = 8080

# Timeout to try to connect with LLM_Server
_C.PROMPTED_MODULE.LLM_CLIENT.TIMEOUT = 60


# Sub Node for LlamaCpp model.
_C.PROMPTED_MODULE.LLAMACPP = CfgNode()

# Path to Model File.
_C.PROMPTED_MODULE.LLAMACPP.MODEL_FILE = "/home/peter/dev/llama/llama-cpp-models/llama-2-13b-chat.gguf.q4_K_M.bin"

# The response temperature.
_C.PROMPTED_MODULE.LLAMACPP.TEMPERATURE = 0.0

# The maximum tokens allowed.
_C.PROMPTED_MODULE.LLAMACPP.MAX_TOKENS = 4096

# The number of transformer layers to add to the GPU. [0 means use CPU]
_C.PROMPTED_MODULE.LLAMACPP.N_GPU_LAYERS = 0

# Whether to stream the LlamaCpp model output to the std out.
_C.PROMPTED_MODULE.LLAMACPP.STREAM_STDOUT = False

# The number of cores to use
_C.PROMPTED_MODULE.LLAMACPP.N_THREADS = 4



def get_cfg() -> CfgNode:
    """
    Return the default config object.
    """
    return _C.clone()