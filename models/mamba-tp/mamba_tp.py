import torch
from mamba_server import server_factory
import argparse

# Parameters
method = "TP"
n_gpus = 2
batch_size = 2
seq_len = 512
EMBEDDING_SIZE = 2560

def mamba_tp(batch_size, seq_len, input_ids=None):

    server = server_factory(method, n_gpus)
    server.prepare()

    if input_ids is None:
        inp = torch.randint(1, int(5e4), size=(batch_size, seq_len, EMBEDDING_SIZE), dtype=torch.long, device='cpu')
    out = server.infer(inp)
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mamba models on multiple GPUs")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-seq_len", type=int, default=1024, help="Sequence length")
    args = parser.parse_args()
    mamba_tp(args.batch_size, args.seq_len)

