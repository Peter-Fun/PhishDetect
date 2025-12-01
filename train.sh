DATE=`date "+%Y-%m-%d_%H-%M"`
CONFIG='lstm'
SPLIT='NONE'
RUN_NAME=${DATE}-${SPLIT}-${CONFIG}

OUTPUT_DIR=runs/$RUN_NAME

# Execute the training command.
python main.py --cfg settings/config_files/$CONFIG.yaml --opts \
    EXECUTION.OUTPUT_DIR $OUTPUT_DIR \
    EXECUTION.MODE train

DATE=`date "+%Y-%m-%d_%H-%M"`
CONFIG='gru'
SPLIT='NONE'
RUN_NAME=${DATE}-${SPLIT}-${CONFIG}

OUTPUT_DIR=runs/$RUN_NAME

# Execute the training command.
python main.py --cfg settings/config_files/$CONFIG.yaml --opts \
    EXECUTION.OUTPUT_DIR $OUTPUT_DIR \
    EXECUTION.MODE train
# Setting up the run

DATE=`date "+%Y-%m-%d_%H-%M"`
CONFIG='transformer'
SPLIT='NONE'
RUN_NAME=${DATE}-${SPLIT}-${CONFIG}

OUTPUT_DIR=runs/$RUN_NAME

# Execute the training command.
python main.py --cfg settings/config_files/$CONFIG.yaml --opts \
    EXECUTION.OUTPUT_DIR $OUTPUT_DIR \
    EXECUTION.MODE train

DATE=`date "+%Y-%m-%d_%H-%M"`
CONFIG='naive_net'
SPLIT='NONE'
RUN_NAME=${DATE}-${SPLIT}-${CONFIG}

OUTPUT_DIR=runs/$RUN_NAME

# Execute the training command.
python main.py --cfg settings/config_files/$CONFIG.yaml --opts \
    EXECUTION.OUTPUT_DIR $OUTPUT_DIR \
    EXECUTION.MODE train
