from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parents[1]
_DATA_DIR = _BASE_DIR / "data"
_MODEL_DIR = _BASE_DIR / "models" / "corgi"


config_corgi = {
    "dna_path": str(_DATA_DIR / "dna_onehot.npy"),
    "bed_path": str(_DATA_DIR / "hg38_sequence_folds_tfexcluded34.bed"),
    "data_dir": str(_DATA_DIR),
    "mask_path": str(_DATA_DIR / "experiment_mask.npy"),
    "trans_regulator_expression_path": str(_DATA_DIR / "tf_expression.npy"),
    "trans_regulator_symbols_path": str(_DATA_DIR / "trans_regulators_final_hgnc.txt"),
    "trans_regulator_ensembl_path": str(_DATA_DIR / "trans_regulators_final_ensg.txt"),
    "experiments_path": str(_DATA_DIR / "experiments_final.txt"),
    "tissue_clusters_path": str(_DATA_DIR / "tissue_clustering.csv"),
    "model_output_path": str(_MODEL_DIR),
    "gnomad_pickle": str(_DATA_DIR / "gnomad_dictionary.pk"),
    
    "seed": 1,
    "batch_size": 2,
    "epochs": 5,
    "warmup_steps": 3000,
    "lr": 1e-4,
    "wd": 1e-3,
    "film_wd": 1e-2,
    "max_runtime": 24 * 3600,               # 1 day
    "safety_margin": 600,                   # 10 mins
    
    'input_trans_regulators':2891,
    'film_mlp_hidden_layers': [512, 256],
    'film_mlp_dropout': 0.2,
    'film_dimensions_conv': [896, 1152],
    'film_dimensions_transformer': [1536, 1536],
    'output_central_bins': 6144,
    'output_channels': 22,
    'dim': 1536,
    'heads': 8,
    'attn_dropout': 0.05,
    'dropout_rate': 0.2,
    'gqa': 2,
    'loss_style': 'adaptive_mn',
    'gradient_clipping': 1.0,
    'checkpoint_every_n': 5000,

    'poisson_loss_weighting': 0.25,
    'loss_epsilon': 1e-5,

    'test_tissues': [46, 47, 49, 50, 54, 105, 159, 160, 161, 174, 202, 203, 211, 212, 213,
                            214, 239, 267, 268, 275, 276, 277, 278, 288, 318, 319, 320, 321,
                            323, 324, 422, 442, 443, 473, 474, 515, 517],
    'valid_tissues': [17, 39, 60, 66, 70, 71, 72, 73, 82, 97, 98, 99, 136, 137, 227,
                            236, 240, 306, 307, 312, 339, 340, 341, 342, 349, 425, 458, 482],
    'easytest_tissues': [3, 25, 64, 87, 90, 98, 106, 115, 124, 137, 159, 189, 192, 247,
                            283, 284, 300, 311, 332, 334, 345, 448, 467, 480, 481, 532, 582],
}