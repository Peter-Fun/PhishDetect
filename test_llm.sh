DATE=`date "+%Y-%m-%d_%H-%M"`
CONFIG='zero_shot_llm'
SPLIT='NONE'
RUN_NAME=${DATE}-${SPLIT}-${CONFIG}

OUTPUT_DIR=runs/$RUN_NAME

# Execute the training command.
python main.py --cfg settings/config_files/$CONFIG.yaml --opts \
    EXECUTION.OUTPUT_DIR $OUTPUT_DIR \
    EXECUTION.MODE test_llm \
    PROMPTED_MODULE.PROMPT_FILE /home/peter/dev/project_sw/PhishDetect/settings/prompts/phish_detect_v2.txt \
    PROMPTED_MODULE.BACKEND LlamaCpp
