from PhishDetect.datasets import build_dataset
from PhishDetect.models import get_model, count_params
from PhishDetect.solver import load_checkpoint
import torch
classes = ["Not Phish", "Phish"]
def demo(cfg):
    """
    Execute the demo!
    """
    train_set = build_dataset(cfg, 'train')
    vocab_size = len( train_set.get_vocab() )
    model = get_model(cfg)(cfg, vocab_size).cuda()
    model, _, _ = load_checkpoint(cfg, model)
    model.train(False)
    tokenizer = train_set.tokenizer
    vocab = train_set.vocab

    user_input = ""
    while user_input.lower() != "quit":
        user_input = input("What action do you want to do? 1: Classify Email Text ")
        if user_input == "1":
            user_input = ""
            inputted = "."
            print("What is the email text?")
            while inputted != "":
                inputted = input(": ")
                user_input += inputted + "\n"
            sequence = torch.tensor(vocab(tokenizer(user_input))).cuda()
            sequence = sequence.unsqueeze(0)
            probs = model(sequence, torch.tensor(len(sequence[0])).unsqueeze(0))
            print(probs)
            preds = torch.argmax(probs.cpu(), 1)
            output = classes[preds.item()]
            print(output)



