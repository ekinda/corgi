import torch
from torch.utils.data import Dataset, Sampler
import random
import numpy as np
import os
import math
import pickle
from .utils import one_hot_to_base, base_to_one_hot

class CorgiSampler(Sampler):
    def __init__(self, sequence_ids, tissue_ids, shuffled=False):
        """
        There is one sample for each combination of sequence and tissue.
        """
        self.sequence_ids = sequence_ids
        self.tissue_ids = tissue_ids
        self.shuffled = shuffled

    def __iter__(self):
        samples = []
        for seq_id in self.sequence_ids:
            for tissue_id in self.tissue_ids:
                samples.append((seq_id, tissue_id))

        if self.shuffled:
            random.shuffle(samples)

        return iter(samples)

    def __len__(self):
        return len(self.sequence_ids) * len(self.tissue_ids)
        
class CorgiDistributedSampler(Sampler):
    """
    A sampler that wraps the CorgiSampler to split the workload among
    distributed processes. It mimics the logic of torch.utils.data.DistributedSampler:
      1. Sets a deterministic ordering (via a seed + epoch).
      2. Pads (or drops) the data to have an even number of samples.
      3. Subsamples the full list according to the process rank.
    """
    def __init__(
        self,
        sequence_ids,
        tissue_ids,
        num_processes: int = 1,
        rank: int = 0,
        seed: int = 0,
        drop_last: bool = False,
        shuffled = False
    ):
        # Create an instance of your base sampler
        self.base_sampler = CorgiSampler(sequence_ids, tissue_ids, shuffled)
        self.dataset_length = len(self.base_sampler)
        if rank >= num_processes or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in [0, {num_processes - 1}]")
        self.num_processes = num_processes
        self.rank = rank
        self.seed = seed
        self.drop_last = drop_last

        # Compute number of samples per replica
        if self.drop_last:
            self.num_samples = self.dataset_length // self.num_processes
        else:
            self.num_samples = int(math.ceil(self.dataset_length / self.num_processes))
        self.total_size = self.num_samples * self.num_processes
        self.epoch = 0

    def __iter__(self):
        # To ensure all replicas use the same ordering, set the numpy random seed
        np.random.seed(self.seed + self.epoch)
        random.seed(self.seed + self.epoch)

        # Get the list of (seq_id, tissue_id) pairs from your base sampler.
        indices = list(self.base_sampler)

        # If not dropping the tail, pad the indices so that total_size is reached.
        if len(indices) < self.total_size:
            padding_size = self.total_size - len(indices)
            # Repeat elements from the beginning to pad.
            indices.extend(indices[:padding_size])
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Subsample: each process gets a slice of the full list.
        indices = indices[self.rank:self.total_size:self.num_processes]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler. When using shuffling, this ensures
        that processes use the same shuffling and thus don't get overlapping data.
        """
        self.epoch = epoch

class CorgiDataset(Dataset):
    def __init__(
        self,
        dna_sequences: str,
        sequence_ids: list,
        tissue_dir: str,
        tissue_ids: list,
        experiment_mask: dict,
        trans_reg_expression: torch.Tensor,
        output_channels: int,
        augment_dna: bool = False,
        augment_gnomad: bool = False,
        augment_trans_reg_std: float = 0,
        gnomad_pickle: str = None,
        trans_reg_clip: tuple = None,
        return_mean_baseline: bool = False,
        mean_baseline_file: str = None,
    ):
        super().__init__()
        self.sequence_ids = sequence_ids
        self.dna_sequences = np.load(dna_sequences, mmap_mode='r')  # shape: (num_seqs, 524288, 4)
        self.tissue_ids = tissue_ids
        self.trans_reg_expression = trans_reg_expression  # shape: (n_tissues, n_trans_regulators)
        self.augment_dna = augment_dna
        self.augment_gnomad = augment_gnomad
        self.augment_trans_reg_std = augment_trans_reg_std
        self.experiment_mask = experiment_mask  # dict: {tissue_id: array([...])} arrays are length 22
        self.output_channels = output_channels

        self.all_labels = {}
        for t_id in self.tissue_ids:
            label_mask = experiment_mask[t_id]
            assert len(label_mask) == output_channels, (
                f"Output channels ({output_channels}) != label_mask length ({len(label_mask)}) for tissue {t_id}"
            )
            tissue_file = os.path.join(tissue_dir, f"tissue_{t_id}.npy")
            if not os.path.isfile(tissue_file):
                raise FileNotFoundError(f"Missing data file for tissue: {tissue_file}")

            tissue_data = np.load(tissue_file, mmap_mode='r')  # shape: (num_seqs, 8192, ~7)
            self.all_labels[t_id] = tissue_data

        self.experiment_mask_torch = {
            key: torch.as_tensor(val).unsqueeze(-1)
            for key, val in self.experiment_mask.items()
        }
        # Optionally clip trans regulator expression values at min and max values.
        if trans_reg_clip is not None:
            clip_min, clip_max = trans_reg_clip
            self.trans_reg_expression = torch.clamp(self.trans_reg_expression, min=clip_min, max=clip_max)

        # Optionally load and prepare the gnomAD augmentations
        self.gnomad_augmentations = {}
        if augment_gnomad:
            self._build_gnomad_augmentations(gnomad_pickle)

        # This is a list of indices where plus and minus strands are flipped.
        # This is needed when you reverse complement a sequence. E.g. cage plus and cage minus should be switched.
        self.rc_flipped_strands = list(range(22))
        self.rc_flipped_strands[12], self.rc_flipped_strands[13] = self.rc_flipped_strands[13], self.rc_flipped_strands[12]
        self.rc_flipped_strands[14], self.rc_flipped_strands[15] = self.rc_flipped_strands[15], self.rc_flipped_strands[14]
        self.rc_flipped_strands[16], self.rc_flipped_strands[17] = self.rc_flipped_strands[17], self.rc_flipped_strands[16]
        self.rc_flipped_strands[18], self.rc_flipped_strands[19] = self.rc_flipped_strands[19], self.rc_flipped_strands[18]

        self.return_mean_baseline = return_mean_baseline
        if return_mean_baseline:
            if mean_baseline_file is None or not os.path.isfile(mean_baseline_file):
                raise ValueError("mean_baseline_file must be provided and exist if return_mean_baseline is True.")
            mb = torch.from_numpy(np.load(mean_baseline_file))            # (n_seq, 8192, C)
            self.mean_baseline = mb.permute(0, 2, 1).contiguous()         # (n_seq, C, 8192)

    def _build_gnomad_augmentations(self, pickle_path):
        """
        Loads the pickled dictionary of differences:
            diff_dict[seq_index][pos] = [ref_base, alt_base]
        Then builds self.gnomad_augmentations[seq_index][pos] = alt_one_hot,
        validating that the reference base in self.dna_sequences matches ref_base.
        """
        print(f"Loading gnomAD pickle: {pickle_path}")
        with open(pickle_path, "rb") as pf:
            diff_dict = pickle.load(pf)

        # For each seq_index in diff_dict, build a subdict of alt alleles in one-hot form
        for seq_index, pos_dict in diff_dict.items():
            # seq_index must map to actual row in self.dna_sequences
            # If your pickled dict uses a different indexing system, adapt accordingly.
            if seq_index >= len(self.dna_sequences):
                # Possibly out of range if there's a mismatch in indexing
                continue

            sub_augment_dict = {}
            for pos_in_interval, (ref_base, alt_base) in pos_dict.items():
                if pos_in_interval < 0 or pos_in_interval >= self.dna_sequences.shape[1]:
                    # Out of range for a single sequence length
                    continue

                # 1) Check that the reference in the dataset matches ref_base
                #    We'll read the one-hot row from self.dna_sequences
                dataset_row = self.dna_sequences[seq_index, pos_in_interval]  # shape (4,)
                dataset_base = one_hot_to_base(dataset_row)
                if dataset_base.encode('utf-8') != ref_base.upper():
                    print(f'Error. Mismatch between ref alleles at sequence {seq_index}, position {pos_in_interval}')
                    continue

                # 2) Build alt one-hot
                alt_one_hot = base_to_one_hot(alt_base)

                # 3) Store in sub_augment_dict
                sub_augment_dict[pos_in_interval] = alt_one_hot

            if len(sub_augment_dict) > 0:
                self.gnomad_augmentations[seq_index] = sub_augment_dict

        print(f"Loaded gnomAD augmentations for {len(self.gnomad_augmentations)} sequences.")

    def __len__(self):
        return len(self.sequence_ids) * len(self.tissue_ids)

    @staticmethod
    def dna_shift(sequence, max_shift=3):
        seq_len = sequence.shape[0]
        shift_val = int(torch.randint(low=-max_shift, high=max_shift+1, size=(1,)).item())
        if shift_val != 0:
            if shift_val > 0:
                sequence = torch.cat([sequence[shift_val:], torch.zeros((shift_val, 4), dtype=sequence.dtype)], dim=0)
            else:
                shift_abs = abs(shift_val)
                sequence = torch.cat([torch.zeros((shift_abs, 4), dtype=sequence.dtype), sequence[:seq_len - shift_abs]], dim=0)
        return sequence

    @staticmethod
    def dna_rc(sequence):
        sequence = torch.flip(sequence, dims=[0])  # Reverse along length
        sequence = sequence[:, [3, 2, 1, 0]]       # Complement: A<->T, C<->G
        return sequence

    def __getitem__(self, index_tuple):
        """
        Args:
            index_tuple (int, int): (sequence_index, tissue_id).
        Returns:
            dna_seq (Tensor) shape (524288, 4)
            trans_reg (Tensor) shape (n_trans_regulators,)
            padded_label (Tensor) shape (output_channels, 8192)
            exp_mask (Tensor) shape (output_channels, 1)

            If using dataloader, a new first dimension will be added for the batch.
        """
        seq_id, tissue_id = index_tuple

        dna_seq = torch.from_numpy(self.dna_sequences[seq_id]).clone()  # shape: (524288, 4)
        trans_reg = self.trans_reg_expression[tissue_id]
        label = self.all_labels[tissue_id][seq_id]  # shape: (8192, n_available_tracks)
        exp_mask = self.experiment_mask_torch[tissue_id]  # shape: (output_channels, 1)

        reverse_aug = False

        # gnomAD-based augmentation BEFORE shifting/RC
        if self.augment_gnomad and seq_id in self.gnomad_augmentations:
            gnomad_dict = self.gnomad_augmentations[seq_id]
            for pos, alt_one_hot in gnomad_dict.items():
                # Toss a coin for each SNP
                if random.random() < 0.5:
                    # Replace reference allele with alt allele
                    dna_seq[pos] = torch.from_numpy(alt_one_hot)

        # Perform shift & reverse-complement
        if self.augment_dna:
            dna_seq = self.dna_shift(dna_seq)
            if random.random() < 0.5:
                dna_seq = self.dna_rc(dna_seq)
                reverse_aug = True

        # Trans regulator expression augmentation
        if self.augment_trans_reg_std:
            noise = torch.rand_like(trans_reg) * self.augment_trans_reg_std
            trans_reg = trans_reg + noise

        # Padding label, flipping if rc
        label = torch.from_numpy(label)
        available_indices = (self.experiment_mask[tissue_id] == 1).nonzero()[0]
        padded_label = torch.zeros((label.shape[0], self.output_channels), dtype=label.dtype)
        padded_label[:, available_indices] = label

        if reverse_aug:
            padded_label = torch.flip(padded_label, dims=[0])
            padded_label = padded_label[:, self.rc_flipped_strands]

        if self.return_mean_baseline:
            mean_baseline = self.mean_baseline[seq_id]  # shape: (C, 8192)
            if reverse_aug:
                mean_baseline = torch.flip(mean_baseline, dims=[-1])
                mean_baseline = mean_baseline[self.rc_flipped_strands, :]
            return dna_seq.type(torch.float16), trans_reg, padded_label.permute(1,0), exp_mask, mean_baseline
        else:
            return dna_seq.type(torch.float16), trans_reg, padded_label.permute(1,0), exp_mask

    def get_item(self, seq_id, tissue_id):
        return self.__getitem__((seq_id, tissue_id))