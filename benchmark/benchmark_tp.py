# benchmark_tp.py
import torch
from pathlib import Path
from models.mamba_tp.mamba_server import server_factory
import argparse

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":  # <- MUST be here
    parser = argparse.ArgumentParser(description="Run Mamba models on multiple GPUs")
    parser.add_argument("-batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("-seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("-n_iter", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    SEQ_LEN = args.seq_len
    EMBEDDING_SIZE = 2560
    N_ITER = args.n_iter

    method = "TP"
    n_gpus = 2

    # Load pre-generated input if exists, else create dummy input
    input_file = Path("input/benchmark_input.pt")
    if input_file.exists():
        input_embeds = torch.load(input_file)
    else:
        input_embeds = torch.randint(
            1, int(5e4), size=(BATCH_SIZE, SEQ_LEN, EMBEDDING_SIZE), dtype=torch.float32, device='cpu'
        )

    # Initialize server and prepare models on GPU
    server = server_factory(method, n_gpus)
    server.prepare()  # Load model once on GPU

    # Move input to server devices
    split_input_embeds = server.split_input(input_embeds)

    # Measure latency and log each cycle with OOM handling
    log_file = LOG_DIR / "tp_latency.log"
    for i in range(N_ITER):
        out = server.infer(input_embeds, counter = i)  # only inference