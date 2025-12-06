import os
import sys
import torch
import time
import importlib.util
import datetime

from PhishDetect.datasets import build_dataset
from PhishDetect.models import get_model, count_params
from PhishDetect.solver import get_optimizer, get_loss_func, load_checkpoint

import common.logging as logging

log = logging.getlogger(__name__)

# Import wandb optionally
spec = importlib.util.find_spec('wandb')
if spec is not None:
    import wandb
    USE_WANDB = True
else:
    USE_WANDB = False
    log.debug(f'Unable to find Weights and Biases installation. Try `pip install wandb`.')

def train(cfg):
    """
    Execute the training loop!
    """

    if USE_WANDB: 
        wandb.init(project="benchmarking", dir=os.path.join(cfg.EXECUTION.OUTPUT_DIR),
            config=cfg, name=os.path.basename(cfg.EXECUTION.OUTPUT_DIR), entity='phish-email'
        )

    # Build the Dataset
    train_set = build_dataset(cfg, 'train')
    test_set = build_dataset(cfg, 'test')

    vocab_size = len( train_set.get_vocab() )
    log.info(f"=== Building Train Data Loader ===")
    train_loader = torch.utils.data.DataLoader(
        train_set, cfg.EXECUTION.BATCH_SIZE, shuffle=True, collate_fn=train_set.collate_fn,
        num_workers=cfg.EXECUTION.NUM_WORKERS
    )
    log.info(f"=== Building Test Data Loader ===")
    test_loader = torch.utils.data.DataLoader(
        test_set, cfg.EXECUTION.BATCH_SIZE, shuffle=True, collate_fn=test_set.collate_fn,
        num_workers=cfg.EXECUTION.NUM_WORKERS
    )

    # Build the Model
    log.info(f"=== Building {cfg.MODEL.ARCHITECTURE.upper()} Model ===")
    model = get_model(cfg)(cfg, vocab_size).cuda()
    num_params = count_params(model)
    log.info(f"The model contains {num_params:,} trainable parameters.")
    model.train()

    # Build the Optimizer
    log.info(f"=== Getting the Optimizer ===")
    optimizer = get_optimizer(cfg, model)

    # Build the Loss Function
    log.info(f"=== Getting the Loss Func. ===")
    criterion = get_loss_func(cfg)

    # Train the Model
    log.info(f"=== Beginning Training ===")
    
    # Load model weights from file
    model, optimizer, start_epoch = load_checkpoint(cfg, model, optimizer)
    
    # Start the training loop
    metrics = {}
    tic = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.TOT_EPOCHS):
        log.info(f"Starting Epoch {epoch + 1}/{cfg.SOLVER.TOT_EPOCHS}")

        # Train an Epoch
        log.info(f"Starting Training ...")
        train_correct = 0.0
        train_class_corr = [0 for _ in range(cfg.HEAD.NUM_CLASSES)]
        train_class_cnts = [0 for _ in range(cfg.HEAD.NUM_CLASSES)]
        train_class_cnts_pres = [0 for _ in range(cfg.HEAD.NUM_CLASSES)]
        cum_train_loss = 0
        model.train(True)
        for i, sample_batch in enumerate(train_loader):
            
            # Split apart and prepare the sample batch
            texts, labels, lengths = sample_batch
            print(texts,labels,lengths)
            texts, labels, lengths = texts.cuda(), labels.cuda(), lengths
            
            # Forward Pass
            optimizer.zero_grad()
            probs = model(texts, lengths)

            # Compute loss
            loss = criterion(probs, labels)

            # Backwards Pass
            loss.backward()
            cum_train_loss += loss.item() * texts.shape[0]

            # Update Weights
            optimizer.step()

            # Calculate the Accuracy
            labels = labels.cpu()
            preds = torch.argmax(probs.cpu(), 1)
            train_correct += sum(preds == labels)

            # Calculate the Recall & Precision
            for ind in range(cfg.HEAD.NUM_CLASSES):
                train_class_corr[ind] += ((labels == ind) * (preds == ind)).sum().item() # True Pos
                train_class_cnts[ind] += (labels == ind).sum().item() # True Pos + False Neg
                train_class_cnts_pres[ind] += (preds == ind).sum().item() # True Pos + False Pos
            torch.cuda.empty_cache()

            # Print Out Performance
            if i % 100 == 0:
                log.info(f'Iteration: {i:>4}/{len(train_loader)} | Loss {loss.item():.4f}')

        # Train an Epoch
        log.info(f"Starting Testing ...")
        test_correct = 0.0
        test_class_corr = [0 for _ in range(cfg.HEAD.NUM_CLASSES)]
        test_class_cnts = [0 for _ in range(cfg.HEAD.NUM_CLASSES)]
        test_class_cnts_pres = [0 for _ in range(cfg.HEAD.NUM_CLASSES)]
        cum_test_loss = 0
        model.train(False)
        for i, sample_batch in enumerate(test_loader):
            # Split apart and prepare the sample batch
            texts, labels, lengths = sample_batch
            texts, labels, lengths = texts.cuda(), labels.cuda(), lengths
            
            # Forward Pass
            probs = model(texts, lengths)

            # Compute loss
            loss = criterion(probs, labels)
            cum_test_loss += loss.item() * texts.shape[0]

            # Calculate the Accuracy
            labels = labels.cpu()
            preds = torch.argmax(probs.cpu(), 1)
            test_correct += sum(preds == labels)

            # Calculate the Recall & Precision
            for ind in range(cfg.HEAD.NUM_CLASSES):
                test_class_corr[ind] += ((labels == ind) * (preds == ind)).sum().item()
                test_class_cnts[ind] += (labels == ind).sum().item()
                test_class_cnts_pres[ind] += (preds == ind).sum().item()
                # print(test_class_corr,test_class_cnts,test_class_cnts_pres)

            torch.cuda.empty_cache()

            # Print Out Performance
            if i % 100 == 0:
                log.info(f'Iteration: {i:>4}/{len(test_loader)} | Loss {loss.item():.4f}')

        toc = time.time()
        elapsed = toc - tic
        sec_per_epoch = elapsed / (epoch + 1)
        remaining = sec_per_epoch * (cfg.SOLVER.TOT_EPOCHS - epoch - 1)

        # Calulating Epoch Metrics
        train_rec = [
            round(train_class_corr[i] / (train_class_cnts[i]+0.00001), 4) 
            for i in range(cfg.HEAD.NUM_CLASSES)
        ]
        test_rec = [
            round(test_class_corr[i] / (test_class_cnts[i]+0.00001), 4)
            for i in range(cfg.HEAD.NUM_CLASSES)
        ]
        train_pres = [
            round(train_class_corr[i] / (train_class_cnts_pres[i]+0.00001), 4) 
            for i in range(cfg.HEAD.NUM_CLASSES)
        ]
        test_pres = [
            round(test_class_corr[i] / (test_class_cnts_pres[i]+0.00001), 4)
            for i in range(cfg.HEAD.NUM_CLASSES)
        ]

        # Form the metrics dictionary.
        metrics = {
            'train/loss': cum_train_loss / len(train_set),
            'test/loss': cum_test_loss / len(test_set),
            'train/acc': train_correct / len(train_set),
            'test/acc': test_correct / len(test_set),
        }
        metrics.update({f'train/recall-{i}': v for i, v in enumerate(train_rec)})
        metrics.update({f'train/precision-{i}': v for i, v in enumerate(train_pres)})
        metrics.update({f'test/recall-{i}': v for i, v in enumerate(test_rec)})
        metrics.update({f'test/precision-{i}': v for i, v in enumerate(test_pres)})

        # Print out the metrics
        metric_str = " | ".join([f'{k} {v}' for k, v in metrics.items()])
        log.info(f'Epoch Summary: Completed {epoch + 1}/{cfg.SOLVER.TOT_EPOCHS} | '
            f'Elapsed {datetime.timedelta(seconds=elapsed)} | '
            f'Remaining {datetime.timedelta(seconds=remaining)}s | '
            f'{metric_str}'
        )
        if USE_WANDB: wandb.log(metrics)

        # Save the model as a checkpoint
        if (epoch + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            checkpoint_path = os.path.join(
                cfg.EXECUTION.OUTPUT_DIR, 'checkpoints', f'chpt_ep{(epoch+1):0>3}.pt'
            )
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                }, checkpoint_path)
