import tqdm
import langchain.llms.llamacpp as llamacpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import common.logging as logging

log = logging.getlogger(__name__)
from PhishDetect.datasets import build_dataset

import LLM.client.prompted_module as pm
import LLM.client.llm_client as llmc

"""
Use a prompted LLM module to understand LLM's zero-shot performance on phishing email detection.

Dfn:
- Zero-Shot Performance: is performance of a model on a task or class that is excluded from the
    training set. 
"""

def test_llm(cfg):
    """
    Testing the llm inference.
    """
    pm.PromptedModule.dev_mode = True

    # Set up the backend for the prompted module.
    if cfg.PROMPTED_MODULE.BACKEND == "LLM_Client":
        model = llmc.LLM_Client(cfg.PROMPTED_MODULE.LLM_CLIENT)
    elif cfg.PROMPTED_MODULE.BACKEND == "LlamaCpp":
        llmcpp_cfg = cfg.PROMPTED_MODULE.LLAMACPP
        model = llamacpp.LlamaCpp(
            model_path=llmcpp_cfg.MODEL_FILE, temperature=llmcpp_cfg.TEMPERATURE, 
            max_tokens=llmcpp_cfg.MAX_TOKENS, n_gpu_layers=llmcpp_cfg.N_GPU_LAYERS,
            n_ctx=llmcpp_cfg.MAX_TOKENS, n_threads=llmcpp_cfg.N_THREADS, top_p=1, n_batch=1024,
            streaming=llmcpp_cfg.STREAM_STDOUT, callbacks=[StreamingStdOutCallbackHandler()],
            verbose=False,
        )
    else:
        NotImplementedError(f'PROMPTED_MODULE.BACKEND {cfg.PROMPTED_MODULE.BACKEND} is not implemented.')

    # Set up the prompted module.
    prompted_module = pm.PromptedModule(cfg.PROMPTED_MODULE, llm_model=model)

    # Set up the test dataset
    test_set = build_dataset(cfg, 'test')
    texts, labels = test_set.texts, test_set.labels

    # # Iterate through dataset
    tp = 0
    fp = 0
    tf = 0
    ff = 0

    for t in tqdm.tqdm(range(len(texts))):
        # Pass the data through the prompted module
        response = prompted_module.ask_llm(texts[t])
        print(response)
        # Interpret the LLM's response to determine if phish
        phish = "is a phishing email" in response
        if phish and labels[t] == 1:
            tp += 1
        elif phish:
            fp += 1
        elif labels[t] == 1:
            ff += 1
        else:
            tf += 1
            
        # keep score on acc, pre, rec
        if t % 50 == 0:
            log.info("Accuracy: " + str((tp+tf)/(tp+fp+tf+ff+0.001)))
            log.info("Phish Precision: " + str((tp)/(tp+fp+0.001)))
            log.info("Phish Recall: " + str((tp)/(tp+ff+0.001)))
            log.info("Not Phish Precision: " + str((tf)/(tf+ff+0.001)))
            log.info("Not Phish Recall: " + str((tf)/(tf+fp+0.001)))
    # # Summarize performance. Look at performance wrt trained PhishDetect transformer.