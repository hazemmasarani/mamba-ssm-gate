import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp 
import argparse

from models.mamba_ssm_gate.mamba_ssm_modeling import MambaForCausalLM_SSM
from models.mamba_ssm_gate.mamba_gate_modeling import MambaForCausalLM_Gate


def setup(rank, world_size, port_num):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port_num)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size, devices, input_ids, port_num, n_iter, return_dict):

    setup(rank, world_size, port_num)
    print(f"Rank {rank} initialized")
    device = torch.device(devices[rank])
    input_ids = input_ids.to(device)

    if rank == 0:
        model = MambaForCausalLM_Gate.from_pretrained("HMasarani/mamba-gate")
    else:
        model = MambaForCausalLM_SSM.from_pretrained("HMasarani/mamba-ssm")

    model = model.to(device)
    model.eval()

    inputs_embeds = input_ids

    seq_len = inputs_embeds.shape[1]

    with torch.no_grad():
        outputs = model(
            inputs_embeds=inputs_embeds,
            use_cache=True
        )

    logits = outputs.logits
    cache = outputs.cache_params

    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    # store tokens on GPU
    tokens = [next_token]

    for step in range(n_iter - 1):

        cache_position = torch.tensor(
            [seq_len + step],
            device=device
        )

        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                cache_params=cache,
                cache_position=cache_position,
                use_cache=True,
                counter=step + 1
            )

        logits = outputs.logits
        cache = outputs.cache_params

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        tokens.append(next_token)

    # concatenate tokens
    tokens = torch.cat(tokens, dim=1).detach().cpu()

    if rank == 0:
        return_dict[rank] = tokens

    cleanup()


def mamba_ssm_gate(dev0, dev1, batch_size, seq_len, port_num, input_ids=None, n_iter=10):

    world_size = 2
    devices = [dev0, dev1]

    # Create input_ids on CPU first
    if input_ids is None:
        print("Creating input_ids on CPU")
        input_ids = torch.randint(
            low=0,
            high=50280,  # Assuming vocab size of 50280, adjust if needed
            size=(batch_size, seq_len)
        )


    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        worker,
        args=(world_size, devices, input_ids, port_num, n_iter, return_dict),
        nprocs=world_size,
        join=True,
    )

    logits_model1 = return_dict[0]
    print("Model logits shape:", logits_model1.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mamba models on multiple GPUs")
    parser.add_argument("-dev0", type=str, required=True, help="First GPU device, e.g. cuda:0")
    parser.add_argument("-dev1", type=str, required=True, help="Second GPU device, e.g. cuda:1")
    parser.add_argument("-batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("-port_num", type=int, default=12355, help="Port number")
    parser.add_argument("-n_iter", type=int, default=10, help="Number of iterations")
    args = parser.parse_args()
    input_embeds = torch.load("input/benchmark_input.pt")
    mamba_ssm_gate(args.dev0, args.dev1, args.batch_size, args.seq_len, args.port_num, input_embeds, args.n_iter)
