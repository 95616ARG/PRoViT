import sys, pathlib
working_dir = pathlib.Path(__file__).parent
sys.path.append(pathlib.Path(__file__).parents[2].as_posix())

import argparse
import torch
from PRoViT.provit import *
import pathlib
import torchvision

parser = argparse.ArgumentParser(description='')
parser.add_argument('--netname', dest='netname', action='store', default='vitb16')
parser.add_argument('--n', type=int, dest='n', action='store', default=5)
parser.add_argument('--method', type=str, dest='method', action='store', required=True, choices=['provitLP', 'provitFT', 'FTall', 'provitFTLP'])
parser.add_argument('--metric', type=int, dest='metric', action='store', default=1)
parser.add_argument('--device', type=str, dest='device', action='store', default='cpu')
parser.add_argument('--dtype', type=str, dest='dtype', action='store', default=torch.float64)
parser.add_argument('--path', type=str, dest='path', action='store', default='/home/public/datasets/ImageNet')
parser.add_argument('--seed', type=int, dest='seed', action='store', default=0)
parser.add_argument('--ft_niter', type=int, default=5)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--gamma', type=float, default=0.995)
args = parser.parse_args()

dtype = args.dtype
device = torch.device(args.device)

""" Load DNN and accuracy """
if args.netname == 'vitl32':
    net, _, _ = models.vit_l_32(dtype=dtype, device=device)   
elif args.netname == 'vitb16':
    net, _, _ = models.vit_b_16(dtype=dtype, device=device)
elif args.netname == 'deit':
    net, _, _ = models.deit(dtype=dtype, device=device)
else:
    raise NotImplementedError

net.eval()
net.netname = args.netname

if isinstance(net, torchvision.models.VisionTransformer):
    last_layer = net.heads[-1]
    reduced_layer = torch.nn.Linear(last_layer.weight.shape[1], 50).to(dtype=dtype, device=device)
    with torch.no_grad():
        reduced_layer.weight[:] = last_layer.weight[:50]
        reduced_layer.bias[:] = last_layer.bias[:50]
    net.heads[-1] = reduced_layer
else: 
    last_layer = net.head
    reduced_layer = torch.nn.Linear(last_layer.weight.shape[1], 50).to(dtype=dtype, device=device)
    with torch.no_grad():
        reduced_layer.weight[:] = last_layer.weight[:50]
        reduced_layer.bias[:] = last_layer.bias[:50]
    net.head = reduced_layer

testset = datasets.get_drawdown_set(dtype=torch.float32, path=args.path, device=device)[:50*50]
if args.netname == 'vitl32':
    og_acc1 = 0.8552
    og_acc5 = 0.9796
elif args.netname == 'deit':
    og_acc1 = 0.8852
    og_acc5 = 0.988
else:
    og_acc1, og_acc5 = testset.accuracy(net, topk=(1, 5))



repair_datasets = datasets.get_repair_sets(int(args.n), metric=args.metric, dtype=dtype, path=args.path, device=device)
gen_datasets = datasets.get_gen_sets(int(args.n), metric=args.metric, dtype=dtype, path=args.path, device=device)

model_dir = working_dir / 'models'
prefix=f"{args.netname}_lastlayer_method={args.method}_n={args.n}_metric={args.metric}_seed={args.seed}"
print(prefix)

""" Compute the original Vision Transformer's accuracy on the generalization sets. """
# Cache Results
cache_gen_acc_dir = working_dir / pathlib.Path(f'cache_gen_acc/net={args.netname}/n={args.n}/metric={args.metric}/seed={args.seed}/dtype={dtype}')
cache_gen_acc_dir.mkdir(parents=True, exist_ok=True)
cache_gen_acc_file = cache_gen_acc_dir / 'cache.pt'

if not cache_gen_acc_file.exists():
    with torch.no_grad():
        torch.save(repair.accuracy_rec(net, gen_datasets), cache_gen_acc_file)

og_gen1, og_gen5, (_, og_gen_dict) = torch.load(cache_gen_acc_file)
print(f"og gen: {og_gen1:.3%}, {og_gen5:.3%}", flush=True)


""" Repair """

model_path = model_dir / f"{prefix}.pt"
argv_path = model_dir / f"{prefix}.argv.pt"
timing_path = model_dir / f"{prefix}.timing.pt"

if model_path.exists():
    if isinstance(net, torchvision.models.VisionTransformer):
        last_layer = net.heads[-1]
        last_layer.load(model_path)
        net.heads[-1] = last_layer
    else:
        last_layer = net.head
        last_layer.load(model_path)
        net.head = last_layer

    timing_dict = torch.load(timing_path)
    encoding_time = timing_dict['encoding T']
    solve_time = timing_dict['solve T']
    ft_time = getattr(timing_dict, 'ft T', 0)

else:
    ft_time = 0
    encoding_time = 0
    solve_time = 0

    if args.method == 'provitFTLP':
        total_encoding_time = 0
        total_ft_time = 0
        total_solve_time = 0

        # Fine tune and repair in a loop until 100% efficacy is guaranteed.
        while True:
            net, last_layer, _, ft_time = repair.fine_tune(
                net, repair_datasets, dtype=dtype, device=device,
                batch_size=args.batch_size,
                optimizer=args.optimizer,
                lr=args.lr,
                gamma=args.gamma,
                max_epoches=args.ft_niter,
                validation_interval=args.ft_niter+1)

            print(f"FT: {ft_time}")
            total_ft_time += ft_time

            net, last_layer, encoding_time, solve_time = repair.repair(
                net, repair_datasets, dtype=dtype, device=device)

            print(f"encoding: {encoding_time}")
            print(f"solve: {solve_time}")
            total_encoding_time += encoding_time
            total_solve_time += solve_time

            # If the repair succeeded, correctness is guaranteed and we are done.
            if last_layer is not None:
                encoding_time = total_encoding_time
                solve_time = total_solve_time
                ft_time = total_ft_time
                break

            import pdb; pdb.set_trace()

    elif args.method == 'provitFT':
        net, last_layer, _, ft_time = repair.fine_tune(
            net, repair_datasets, dtype=dtype, device=device,
            batch_size=10, optimizer='Adam', lr=1e-3, gamma=0.995, only_last_layer=True)

    elif args.method == 'FTall':
        net, last_layer, _, ft_time = repair.fine_tune(
            net, repair_datasets, dtype=dtype, device=device,
            batch_size=10, optimizer='Adam', lr=1e-3, gamma=0.995, only_last_layer=False)

    elif args.method == 'provitLP':
        net, last_layer, encoding_time, solve_time = repair.repair(net, repair_datasets, dtype=dtype, device=device, eps=0.1)

    else:
        raise NotImplementedError

    print(f"time: {encoding_time:.1f}s + {solve_time:.1f}s + {ft_time:.1f}s")

""" Preparation for Evaluation. """
dtype = torch.float32

""" Evaluate Drawdown """
# Compute efficacy, drawdown, and generalization.
with torch.no_grad():
    net = net.to(dtype=torch.float32)
    efficacy = repair.accuracy_rec(net, [d.to(dtype=torch.float32) for d in repair_datasets])[0]
    print(f"efficacy: {efficacy:.3%}", flush=True)
    assert(efficacy == 1.0)
    acc1, acc5 = testset.accuracy(net, topk=(1, 5))
    print(f"acc: {acc1:.3%}, {acc5:.3%}", flush=True)
    print(f"D: {float(og_acc1 - acc1):.3%}, {float(og_acc5 - acc5):.3%}", flush=True)
    new_gen1, new_gen5, (_, new_gen_dict) = repair.accuracy_rec(net, gen_datasets)
    print(f"gen: {new_gen1:.3%}, {new_gen5:.3%}", flush=True)

# Log results.
result = {
    'E@top-1': efficacy,
    'D@top-1': float(og_acc1 - acc1),
    'D@top-5': float(og_acc5 - acc5),
    'G@top-1': new_gen1 - og_gen1,
    'G@top-5': new_gen5 - og_gen5,
    'encoding T' : encoding_time,
    'solve T': solve_time,
    'ft T': ft_time,
    'OG Gen1': og_gen1,
    'OG Gen5': og_gen5,
    'New Gen1': new_gen1,
    'New Gen5': new_gen5,
    'OG Gen Dict': og_gen_dict,
    'New Gen Dict': new_gen_dict,
}

torch.save(result, model_dir / f"{prefix}.result.pt")

with open(working_dir / 'results' / f"{prefix}.result.txt", 'a') as file:
    file.write(f"N = {args.n}\n")
    for key, val in result.items():
        if not isinstance(val, dict):
            print(f"{key}: {val}")
        file.write(f"{key}: {val}\n")
    file.write("\n")