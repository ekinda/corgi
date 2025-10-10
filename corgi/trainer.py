import os
import time
import datetime
import psutil
import logging
import numpy as np
import pandas as pd
from socket import gethostname

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .model import Corgi
from .data_classes import CorgiDataset, CorgiDistributedSampler
from .utils import load_experiment_mask, poisson_multinomial_masked_v2

class CorgiBaseTrainer:
    def __init__(self, config):
        """
        Initializes the trainer with the given configuration.
        """
        self.config = config
        self.global_step = 0
        self.start_epoch = 0
        self.start_time = time.time()

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            raise NotImplementedError("CUDA not found. CPU support is not implemented. Exiting.")

    @staticmethod
    def parse_bed_file(bed_path):
        """Parse the BED file to split sequences into train/valid/test indices."""
        df = pd.read_csv(bed_path, sep='\t', header=None, names=["chr", "start", "end", "fold"])
        test_indices = df.index[df["fold"] == "fold3"].tolist()
        valid_indices = df.index[df["fold"] == "fold4"].tolist()
        train_indices = df.index[~df["fold"].isin(["fold3", "fold4"])].tolist()
        return train_indices, valid_indices, test_indices
    
    @staticmethod
    def crop_tensor(tensor, target_length):
        current_length = tensor.shape[-1]
        if current_length == target_length:
            return tensor
        elif current_length < target_length:
            raise ValueError("Tensor length is smaller than target cropping length.")
        crop_amount = (current_length - target_length) // 2
        return tensor[..., crop_amount:crop_amount + target_length]
        
    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def _init_ddp(self):
        self.rank          = int(os.environ["SLURM_PROCID"])
        self.world_size    = int(os.environ["WORLD_SIZE"])
        self.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        assert self.gpus_per_node == torch.cuda.device_count()

        print(f"Hello from rank {self.rank} of {self.world_size} on {gethostname()} where there are" \
            f" {self.gpus_per_node} allocated GPUs per node.", flush=True)

        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size,
                                timeout=datetime.timedelta(seconds=600))

        if self.rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

        self.local_rank = self.rank - self.gpus_per_node * (self.rank // self.gpus_per_node)
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)

    def _prepare_data(self):
        """
        Loads genome data, experiment mask and expression data;
        splits sequences and tissues into training/validation sets;
        and fixes evaluation samples.
        """
        cfg = self.config
        
        tissue_clusters = pd.read_csv(cfg['tissue_clusters_path'])

        self.valid_tissues = cfg['valid_tissues']
        self.test_tissues = cfg['test_tissues']
        self.easy_test_tissues = cfg['easytest_tissues']
        self.train_tissues = [x for x in tissue_clusters.tissue_id.values
                              if x not in self.valid_tissues + self.test_tissues + self.easy_test_tissues]

        # Split sequences via BED file
        train_seq_idx, valid_seq_idx, test_seq_idx = self.parse_bed_file(cfg["bed_path"])
        self.valid_seq_idx = valid_seq_idx
        self.train_seq_idx = train_seq_idx
        self.test_seq_idx = test_seq_idx

        if self.rank == 0:
            logging.info(f'Training sequences: {len(train_seq_idx)}, validation sequences: {len(valid_seq_idx)}, test sequences: {len(test_seq_idx)}')
            logging.info(f'Training tissues: {len(self.train_tissues)}')
            logging.info(f'Validation tissues: {len(self.valid_tissues)}')
            logging.info(f'Easy test tissues: {len(self.easy_test_tissues)}')
            logging.info(f'Test tissues: {len(self.test_tissues)}')

        # Load genome data and auxiliary files.
        self.dna_path = cfg["dna_path"]
        self.experiment_mask = load_experiment_mask(cfg["mask_path"])
        self.trans_reg_expression = torch.from_numpy(np.load(cfg["trans_regulator_expression_path"])).float()
            
    def _build_optimizer_scheduler(self):
        """
        Sets up the optimizer with different LRs for LoRA vs. other parameters,
        and configures a OneCycleLR scheduler and gradient scaler.
        """
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
            {"params": film_mlp_params, "lr": cfg["lr"], "weight_decay": cfg["film_wd"]},
            {"params": other_params, "lr": cfg["lr"], "weight_decay": cfg["wd"]}
        ]
    )
        # Compute total steps (this formula can be adjusted)
        total_steps = int(np.ceil((cfg["epochs"] * len(self.train_seq_idx) * len(self.train_tissues)) / (self.world_size * cfg['batch_size'])))
        if self.rank == 0:
            logging.info(f"Total steps: {total_steps}")

        warmup_scheduler = LinearLR(self.optimizer, start_factor = 0.000001, total_iters = cfg['warmup_steps'])
        main_scheduler  = CosineAnnealingLR(self.optimizer, T_max=total_steps - cfg['warmup_steps'], eta_min=0.0)
        self.scheduler = SequentialLR(self.optimizer, [warmup_scheduler, main_scheduler], [cfg['warmup_steps']])

    def save_checkpoint(self, epoch):
        """
        Saves a checkpoint with model, optimizer, scaler, and scheduler states.
        """
        if self.rank == 0:
            cfg = self.config
            model_to_save = self.model.module if isinstance(self.model, DDP) else self.model
            timestamp = time.strftime('%Y-%m-%d_%H:%M', time.localtime())
            checkpoint_path = os.path.join(cfg["model_output_path"], f"grt_epoch_{epoch}_{timestamp}.pt")

            torch.save({
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}.")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Loads checkpoint state if a resume checkpoint path is provided.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])
        
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        if self.rank == 0:
            logging.info(f"Resumed from checkpoint at epoch {self.start_epoch}, global step {self.global_step}.")

    def load_checkpoint_modelonly(self, checkpoint_path):
        """
        Loads checkpoint state if a resume checkpoint path is provided.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt["model_state_dict"])

        if self.rank == 0:
            logging.info(f"Resumed from model checkpoint at {checkpoint_path}")

class CorgiTrainer(CorgiBaseTrainer):
    def __init__(self, config):
        super(CorgiTrainer, self).__init__(config)

        self._init_ddp()
        self._set_seed(self.config["seed"])
        self._prepare_data()
        self._build_model()
        self._build_optimizer_scheduler()

    def _build_model(self):
        model = Corgi(self.config)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(self.local_rank)
        model = DDP(model, device_ids = [self.local_rank])
        self.model = model

        if self.rank == 0:
            logging.info(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    def train(self):
        """
        Main training loop over epochs, tissue chunks and batches.
        """
        cfg = self.config
        loss_epsilon = cfg['loss_epsilon']
        should_stop = False

        if cfg['loss_style'] in ['adaptive_mn']:                    # Other loss styles are not currently supported
            poisson_loss_weight = cfg['poisson_loss_weighting']

        train_dataset = CorgiDataset(
            dna_sequences=cfg['dna_path'],
            sequence_ids=self.train_seq_idx,
            tissue_dir=cfg["data_dir"],
            tissue_ids=self.train_tissues,
            experiment_mask=self.experiment_mask,
            trans_reg_expression=self.trans_reg_expression,
            output_channels=cfg["output_channels"],
            augment_dna=True,
            augment_gnomad=True,
            augment_trans_reg_std=0.02,
            gnomad_pickle=cfg["gnomad_pickle"],
            trans_reg_clip=None
        )
        sampler = CorgiDistributedSampler(
            sequence_ids=self.train_seq_idx,
            tissue_ids=self.train_tissues,
            num_processes = self.world_size,
            rank = self.rank,
            seed = self.seed,
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
                dna_seq, trans_reg, label, exp_mask = batch
                dna_seq = dna_seq.to(self.local_rank, non_blocking=True)
                trans_reg = trans_reg.to(self.local_rank, non_blocking=True)
                label = label.to(self.local_rank, non_blocking=True)
                exp_mask = exp_mask.to(self.local_rank, non_blocking=True)

                self.optimizer.zero_grad()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(dna_seq, trans_reg)
                    cropped_label = self.crop_tensor(label, self.config["output_central_bins"])
                    channel_losses = poisson_multinomial_masked_v2(outputs, cropped_label, exp_mask, poisson_loss_weight, loss_epsilon)
                    weights = (1 / (2 * self.model.module.loss_channel_weights ** 2))  # shape: (22,)
                    channel_losses = (channel_losses * weights.unsqueeze(0) + torch.log(self.model.module.loss_channel_weights).unsqueeze(0)) * exp_mask.squeeze(-1)
                    loss = channel_losses.sum() / exp_mask.sum()
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clipping'])
                self.optimizer.step()
                self.scheduler.step()

                if self.rank == 0:
                    if step % 1000 == 0:
                        current_lr = self.scheduler.get_last_lr()[0]
                        allocated_mem = torch.cuda.memory_allocated(self.local_rank) / 1e9  # in GB
                        reserved_mem = torch.cuda.memory_reserved(self.local_rank) / 1e9    # in GB
                        logging.info(f"Epoch {epoch} Step {step} loss: {loss.item():.4f}, lr: {current_lr:.4E}, "
                                     f"GPU Allocated: {allocated_mem:.2f} GB, GPU Reserved: {reserved_mem:.2f} GB, "
                                     f"virtual mem: {psutil.virtual_memory().used / 1e9:.1f} GB")
                now = time.time()
                if now + cfg['safety_margin'] >= self.start_time + cfg['max_runtime']:
                    logging.info("Max runtime reached; final evaluation and checkpointing.")
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