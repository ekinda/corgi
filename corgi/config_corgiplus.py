from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parents[1]
_DATA_REV = Path('/project/deeprna/data/revision')
_DNASE_DIR = _DATA_REV / 'dnase'
_MODEL_DIR = '/project/deeprna/models/corgi_revision/tests/'

# Shared defaults; override in scripts/run_corgiplus_local.py as needed
config_corgiplus = {
    # core data
    'dna_path': str(Path('/project/deeprna_data/pretraining_data_final2/dna_onehot.npy')),
    'bed_path': str(Path('/project/deeprna_data/pretraining_data_final2/hg38_sequence_folds_tfexcluded34.bed')),
    'tissue_dir': str(Path('/project/deeprna_data/pretraining_data_final2')),
    'mask_path': str(Path('/project/deeprna_data/pretraining_data_final2/experiment_mask.npy')),
    'experiments_path': str(Path('/project/deeprna_data/pretraining_data_final2/experiments_final.txt')),
    'trans_regulator_expression_path': str(Path('/project/deeprna_data/pretraining_data_final2/tf_expression.npy')),
    'gnomad_pickle': str(Path('/project/deeprna_data/pretraining_data_final2/gnomad_dictionary.pk')),
    'dnase_global_path': str(_DNASE_DIR / 'hv_marker_embeddings_z.npy'),
    'hv_marker_bed': str(_DNASE_DIR / 'hv_marker_reference.bed'),
    'training_tissues_path': str(_DATA_REV / 'training_samples.txt'),
    'validation_tissues_path': str(_DATA_REV / 'validation_samples.txt'),
    'test_tissues_path': str(_DATA_REV / 'test_samples.txt'),

    # model dims
    'output_channels': 22,
    'dim': 1536,
    'film_dimensions_conv': [896, 1152],
    'film_dimensions_transformer': [1536, 1536],
    'film_mlp_hidden_layers': [512, 256],
    'film_mlp_dropout': 0.2,
    'input_trans_regulators': 2891,
    'output_central_bins': 6144,
    'heads': 8,
    'attn_dropout': 0.05,
    'dropout_rate': 0.2,
    'gqa': 2,
    'loss_style': 'adaptive_mn',
    'loss_epsilon': 1e-5,
    'poisson_loss_weighting': 0.25,

    # aux defaults
    'corgiplus_aux_hidden_dim': 128,
    'corgiplus_aux_out_dim': 256,
    'corgiplus_aux_dropout': 0.1,

    # training
    'batch_size': 1,
    'epochs': 1,
    'lr': 1e-4,
    'wd': 1e-3,
    'film_wd': 1e-2,
    'warmup_steps': 100,
    'max_runtime': 24 * 3600,
    'safety_margin': 600,
    'checkpoint_every_n': 100,
    'gradient_clipping': 1.0,
    'seed': 1,
    'model_output_path': str(_MODEL_DIR),
    'finetune_checkpoint': str(Path('/project/deeprna_data/corgi-reproduction/data/corgi_model.pt')),
}
