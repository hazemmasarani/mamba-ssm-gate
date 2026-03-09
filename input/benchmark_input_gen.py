import torch
import argparse

EMBEDDING_SIZE = 2560
VOCAB_SIZE = 50280

def generate_input(batch_size, seq_len):
    inp = torch.randint(1, int(5e4), size=(batch_size, seq_len, EMBEDDING_SIZE), device='cpu')
    print(f"Input shape: {inp.shape}")
    torch.save(inp, "input/benchmark_input.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mamba models on multiple GPUs")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-seq_len", type=int, default=1024, help="Sequence length")
    args = parser.parse_args()
    generate_input(args.batch_size, args.seq_len)