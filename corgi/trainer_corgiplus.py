import os
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from socket import gethostname
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .model import Corgi, CorgiPlus, CorgiPlusNofilm
from .data_classes import CorgiDataset, CorgiDistributedSampler
from .trainer import CorgiBaseTrainer
from .utils import load_experiment_mask, poisson_multinomial_masked_v2

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')


class CorgiPlusDataset(CorgiDataset):
    """Return tissue ids alongside tensors for DNase conditioning."""

    def __getitem__(self, index_tuple):
        dna_seq, trans_reg, padded_label, exp_mask = super().__getitem__(index_tuple)
        _, tissue_id = index_tuple
        return dna_seq, trans_reg, padded_label, exp_mask, tissue_id


def _load_ids(path: str) -> List[int]:
    with open(path) as f:
        return [int(x) for x in f.read().strip().split() if x.strip()]


class CorgiPlusTrainer(CorgiBaseTrainer):
    """DDP trainer for CorgiPlus and fine-tuning variants."""

    def __init__(self, config, mode: str = 'rna', training_mode: str = 'slurm'):
        self.mode = mode
        super().__init__(config, training_mode=training_mode)
        self._init_ddp()
        self._set_seed(self.config["seed"])
        self._prepare_data()
        self._build_model()
        self._build_optimizer_scheduler()

    @staticmethod
    def parse_bed_file(bed_path):
        """Parse BED splits for revised data (fold3=test, fold4=valid)."""
        df = pd.read_csv(bed_path, sep='\t', header=None, names=["chr", "start", "end", "fold"])
        test_indices = df.index[df["fold"] == "fold3"].tolist()
        valid_indices = df.index[df["fold"] == "fold4"].tolist()
        train_indices = df.index[~df["fold"].isin(["fold3", "fold4"])].tolist()
        return train_indices, valid_indices, test_indices

    def _prepare_data(self):
        cfg = self.config

        self.train_tissues = _load_ids(cfg['training_tissues_path'])
        self.valid_tissues = _load_ids(cfg['validation_tissues_path'])
        self.test_tissues = _load_ids(cfg['test_tissues_path'])

        train_seq_idx, valid_seq_idx, test_seq_idx = self.parse_bed_file(cfg["bed_path"])
        self.valid_seq_idx = valid_seq_idx
        self.train_seq_idx = train_seq_idx
        self.test_seq_idx = test_seq_idx

        if self.rank == 0:
            logging.info(f'Training sequences: {len(train_seq_idx)}, validation sequences: {len(valid_seq_idx)}, test sequences: {len(test_seq_idx)}')
            logging.info(f'Training tissues: {len(self.train_tissues)}')
            logging.info(f'Validation tissues: {len(self.valid_tissues)}')
            logging.info(f'Test tissues: {len(self.test_tissues)}')

        self.dna_path = cfg["dna_path"]
        self.experiment_mask = load_experiment_mask(cfg["mask_path"])
        self.trans_reg_expression = torch.from_numpy(np.load(cfg["trans_regulator_expression_path"])).float()

        self.dnase_global = None
        if self.mode in ("dnase", "dnase_nofilm"):
            self.dnase_global = torch.from_numpy(np.load(cfg['dnase_global_path'])).float().to(self.device)

    def _build_model(self):
        cfg = dict(self.config)

        if self.mode in ("dnase", "dnase_nofilm") and self.dnase_global is not None:
            cfg['input_trans_regulators'] = self.dnase_global.shape[1]
            cfg['corgiplus_aux_input_dim'] = 1
        elif self.mode == 'rna':
            cfg['corgiplus_aux_input_dim'] = 2

        if self.mode == 'rna':
            model = CorgiPlus(cfg)
        elif self.mode == 'dnase':
            model = CorgiPlus(cfg)
        elif self.mode == 'dnase_nofilm':
            model = CorgiPlusNofilm(cfg)
        elif self.mode in ('corgi_finetune_gnomad', 'corgi_finetune_nognomad'):
            model = Corgi(cfg)
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

        if self.mode in ('corgi_finetune_gnomad', 'corgi_finetune_nognomad'):
            ckpt_path = cfg.get('finetune_checkpoint')
            if ckpt_path and Path(ckpt_path).exists():
                state = torch.load(ckpt_path, map_location='cpu')
                state_dict = state.get('model_state_dict', state)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if self.rank == 0:
                    logging.info(f"Loaded checkpoint {ckpt_path}; missing={len(missing)}, unexpected={len(unexpected)}")

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(self.local_rank)
        if self.training_mode == 'slurm':
            model = DDP(model, device_ids=[self.local_rank])
        self.model = model

        if self.rank == 0:
            logging.info(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    def _build_optimizer_scheduler(self):
        cfg = self.config
        film_mlp_params, other_params = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "film_mlp" in name:
                    film_mlp_params.append(param)
                else:
                    other_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": film_mlp_params, "lr": cfg["lr"], "weight_decay": cfg.get("film_wd", cfg['wd'])},
                {"params": other_params, "lr": cfg["lr"], "weight_decay": cfg['wd']}
            ]
        )

        total_steps = int(np.ceil((cfg["epochs"] * len(self.train_seq_idx) * len(self.train_tissues)) / (self.world_size * cfg['batch_size'])))
        if self.rank == 0:
            logging.info(f"Total steps: {total_steps}")

        warmup_scheduler = LinearLR(self.optimizer, start_factor=0.000001, total_iters=cfg['warmup_steps'])
        main_scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, total_steps - cfg['warmup_steps']), eta_min=0.0)
        self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, main_scheduler], [cfg['warmup_steps']])

    def save_checkpoint(self, epoch):
        """
        Saves a checkpoint with model, optimizer, scaler, and scheduler states.
        """
        if self.rank == 0:
            cfg = self.config
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            timestamp = time.strftime('%Y-%m-%d_%H:%M', time.localtime())
            checkpoint_path = os.path.join(cfg["model_output_path"], f"corgiplus_{self.mode}_epoch_{epoch}_{timestamp}.pt")

            torch.save({
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}.")

    def train(self):
        cfg = self.config
        loss_epsilon = cfg['loss_epsilon']
        should_stop = False

        if cfg['loss_style'] in ['adaptive_mn']:
            poisson_loss_weight = cfg['poisson_loss_weighting']
        else:
            poisson_loss_weight = 1.0

        augment_dna = cfg.get('augment_dna', True)
        augment_gnomad = cfg.get('augment_gnomad', False) or self.mode == 'corgi_finetune_gnomad'
        augment_tr_std = cfg.get('augment_trans_reg_std', 0.02)

        train_dataset = CorgiPlusDataset(
            dna_sequences=cfg['dna_path'],
            sequence_ids=self.train_seq_idx,
            tissue_dir=cfg['tissue_dir'],
            tissue_ids=self.train_tissues,
            experiment_mask=self.experiment_mask,
            trans_reg_expression=self.trans_reg_expression,
            output_channels=cfg['output_channels'],
            augment_dna=augment_dna,
            augment_gnomad=augment_gnomad,
            augment_trans_reg_std=augment_tr_std,
            gnomad_pickle=cfg.get('gnomad_pickle'),
            trans_reg_clip=None
        )

        sampler = CorgiDistributedSampler(
            sequence_ids=self.train_seq_idx,
            tissue_ids=self.train_tissues,
            num_processes=self.world_size,
            rank=self.rank,
            seed=self.seed,
            shuffled=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            sampler=sampler,
            num_workers=4,
            persistent_workers=False,
            pin_memory=False
        )

        for epoch in range(self.start_epoch, cfg["epochs"]):
            if self.rank == 0:
                logging.info(f"--- Epoch {epoch}/{cfg['epochs']} ---")
            step = 0
            sampler.set_epoch(epoch)

            for batch in train_loader:
                dna_seq, trans_reg, label, exp_mask, tissue_id = batch
                dna_seq = dna_seq.to(self.local_rank, non_blocking=True)
                trans_reg = trans_reg.to(self.local_rank, non_blocking=True)
                label = label.to(self.local_rank, non_blocking=True)
                exp_mask = exp_mask.to(self.local_rank, non_blocking=True)
                tissue_id = tissue_id.to(self.local_rank, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    masked_exp = exp_mask.clone()

                    if self.mode == 'rna':
                        aux = label[:, 16:18, :]
                        masked_exp[:, 16:18, :] = 0
                        cond = trans_reg
                        outputs = self.model(dna_seq, aux.permute(0, 2, 1), cond)
                    elif self.mode == 'dnase':
                        aux = label[:, 0:1, :]
                        masked_exp[:, 0:1, :] = 0
                        cond = self.dnase_global[tissue_id]
                        outputs = self.model(dna_seq, aux.permute(0, 2, 1), cond)
                    elif self.mode == 'dnase_nofilm':
                        aux = label[:, 0:1, :]
                        masked_exp[:, 0:1, :] = 0
                        outputs = self.model(dna_seq, aux.permute(0, 2, 1), trans_reg=None)
                    elif self.mode in ('corgi_finetune_gnomad', 'corgi_finetune_nognomad'):
                        outputs = self.model(dna_seq, trans_reg)
                    else:
                        raise ValueError(self.mode)

                    cropped_label = self.crop_tensor(label, cfg["output_central_bins"])
                    channel_losses = poisson_multinomial_masked_v2(outputs, cropped_label, masked_exp, poisson_loss_weight, loss_epsilon)

                    model_ref = self.model.module if isinstance(self.model, DDP) else self.model

                    if hasattr(model_ref, 'loss_channel_weights'):
                        weights = (1 / (2 * model_ref.loss_channel_weights ** 2))
                        channel_losses = (channel_losses * weights.unsqueeze(0) + torch.log(model_ref.loss_channel_weights).unsqueeze(0)) * masked_exp.squeeze(-1)
                        loss = channel_losses.sum() / masked_exp.sum()
                    else:
                        loss = (channel_losses * masked_exp.squeeze(-1)).sum() / masked_exp.sum()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
                self.optimizer.step()
                self.scheduler.step()

                if self.rank == 0 and step % 1000 == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    allocated_mem = torch.cuda.memory_allocated(self.local_rank) / 1e9
                    reserved_mem = torch.cuda.memory_reserved(self.local_rank) / 1e9
                    logging.info(
                        f"Epoch {epoch} Step {step} loss: {loss.item():.4f}, lr: {current_lr:.4E}, "
                        f"GPU Allocated: {allocated_mem:.2f} GB, GPU Reserved: {reserved_mem:.2f} GB"
                    )

                now = time.time()
                if now + cfg['safety_margin'] >= self.start_time + cfg['max_runtime']:
                    logging.info("Max runtime reached; checkpointing.")
                    self.save_checkpoint(epoch)
                    should_stop = True
                    break

                elif step % cfg['checkpoint_every_n'] == 0:
                    self.save_checkpoint(epoch)

                self.global_step += 1
                step += 1

            if should_stop:
                break

            self.save_checkpoint(epoch)

        logging.info("Training complete, exiting.")
