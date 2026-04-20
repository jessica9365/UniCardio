import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import os
import random
import csv
import torch.nn.functional as F
class SimpleCSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        # If the file doesn't exist, create one with headers.
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['epoch', 'metric1', 'value1', 'stage', 'threshold', 'lr'])

    def log(self, epoch, metric1, value1, stage, threshold, lr):
        with open(self.filepath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([epoch, metric1, value1, stage, threshold, lr])
class thresholder:
    @staticmethod
    def forward(batch_number):
        if  batch_number <= 200:
            stage = 1 
            threshold = 0
        elif batch_number > 200 and batch_number <= 250:
            stage = 2
            threshold = 0.5
        elif batch_number > 250 and batch_number <= 350:
            stage = 2
            threshold = 0.5
        elif batch_number > 350 and batch_number <= 400:
            stage = 2
            threshold = 0.5
        elif batch_number > 400 and batch_number <=450:
            stage = 3 
            threshold = 0.5
        elif batch_number > 450 and batch_number <=550:
            stage = 3 
            threshold = 0.5
        elif batch_number > 550 and batch_number <=600:
            stage = 3 
            threshold = 0.5
        elif batch_number > 600 and batch_number <=800:
            stage = 3 
            threshold = 2/3
        return stage, threshold
            
def train(
        model,
        config,
        train_loader,
        valid_loader=None,
        valid_epoch_interval=10,
        foldername="",
        device='cuda',
    ):
        device = device if isinstance(device, torch.device) else torch.device(device)
        model.to(device)

        # --- Option A: freeze backbone, train SBP/DBP head only ---
        for p in model.parameters():
            p.requires_grad = False

        # Access underlying diff_CSDI inside CSDI_base (and DataParallel)
        core = model.module if isinstance(model, torch.nn.DataParallel) else model
        diff_model = core.diffmodel   # adjust if your CSDI_base uses a different attribute name

        for p in diff_model.sbp_dbp_head.parameters():
            p.requires_grad = True

        optimizer = Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            weight_decay=1e-6
        )

        foldername = "./check"
        if foldername != "":
            os.makedirs(foldername, exist_ok=True)
            log_path = os.path.join(foldername, 'loss.csv')
            logger = SimpleCSVLogger(log_path)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[150, 200], gamma=0.1
        )

        num_epochs = 200

        for epoch_no in range(num_epochs):
            avg_loss = 0.0
            model.train()

            with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, train_batch in enumerate(it, start=1):
                    # train_batch = (signal, impute, noisy, mask, bp_vals, subject_id)
                    signal   = train_batch[0].to(device)      # (B, 2000)
                    mask     = train_batch[3].to(device)      # (B, 500)
                    bp_vals  = train_batch[4].to(device)      # (B, 2) SBP, DBP mmHg

                    # For head-only fine-tune we can feed the clean signal as "noisy"
                    noisy = signal

                    # diffusion_step: use a fixed small step or random; here fixed 0
                    diffusion_step = torch.zeros(signal.size(0), dtype=torch.long, device=device)

                    optimizer.zero_grad()

                    # mode=3, borrow_mode=2 for ECG+PPG -> ABP in your packing
                    pred_wave, bp_pred = model(
                        noisy,
                        diffusion_step,
                        mask,
                        mode=3,
                        borrow_mode=2,
                        return_bp_values=True
                    )

                    # BP-only loss (Option A)
                    bp_loss = F.l1_loss(bp_pred, bp_vals) + F.mse_loss(bp_pred, bp_vals)
                    loss = bp_loss

                    loss.backward()
                    optimizer.step()

                    avg_loss += loss.item() / signal.size(0)

                    it.set_postfix(
                        ordered_dict={
                            "avg_epoch_loss": avg_loss / batch_no,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )

                    if batch_no >= config["itr_per_epoch"]:
                        break

            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]
            logger.log(epoch_no, 'bp_loss', avg_loss, stage=1, threshold=0.0, lr=current_lr)

            if foldername != "":
                output_path = os.path.join(foldername, f'model_fine_{epoch_no}.pth')
                torch.save(model.state_dict(), output_path)