#!/bin/bash

python input/benchmark_input_gen.py -batch_size 1 -seq_len 512
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 1 -seq_len 512 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 1 -seq_len 512 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20


python input/benchmark_input_gen.py -batch_size 1 -seq_len 256
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 1 -seq_len 256 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 1 -seq_len 256 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20

python input/benchmark_input_gen.py -batch_size 1 -seq_len 128
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 1 -seq_len 128 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 1 -seq_len 128 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20

python input/benchmark_input_gen.py -batch_size 1 -seq_len 64
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 1 -seq_len 64 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 1 -seq_len 64 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20

python input/benchmark_input_gen.py -batch_size 2 -seq_len 256
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 2 -seq_len 256 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 2 -seq_len 256 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20

python input/benchmark_input_gen.py -batch_size 2 -seq_len 128
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 2 -seq_len 128 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 2 -seq_len 128 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20

python input/benchmark_input_gen.py -batch_size 2 -seq_len 64
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 2 -seq_len 64 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 2 -seq_len 64 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20


python input/benchmark_input_gen.py -batch_size 4 -seq_len 128
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 4 -seq_len 128 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 4 -seq_len 128 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20

python input/benchmark_input_gen.py -batch_size 4 -seq_len 64
PYTHONPATH=. python benchmark/benchmark_tp.py -batch_size 4 -seq_len 64 -n_iter 20
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 4 -seq_len 64 -dev0 cuda:2 -dev1 cuda:3 -n_iter 20

python input/benchmark_input_gen.py -batch_size 1 -seq_len 1024
PYTHONPATH=. python benchmark/benchmark_ssm_gate.py -batch_size 1 -seq_len 1024 -dev0 cuda:2 -dev1 cuda:3 -n_iter 2