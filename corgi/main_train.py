from trainer import CorgiTrainer
from config import config_corgi
import logging
import torch.distributed as dist

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    trainer = CorgiTrainer(config_corgi)
    trainer.train()
    
    dist.destroy_process_group()