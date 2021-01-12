import os
import time
import argparse

# TODO please configure TORCH_HOME and data_paths before running
TORCH_HOME = "/ssd1/chenwy"  # Path that contains the nas-bench-201 database. If you only want to run on NASNET (i.e. DARTS) search space, then just leave it empty
data_paths = {
    "cifar10": "/ssd1/cifar.python",
    "cifar100": "/ssd1/cifar.python",
    "ImageNet16-120": "/ssd1/ImageNet16",
    "imagenet-1k": "/ssd2/chenwy/imagenet_final",
}


parser = argparse.ArgumentParser("TENAS_launch")
parser.add_argument('--gpu', default=0, type=int, help='use gpu with cuda number')
parser.add_argument('--space', default='nas-bench-201', type=str, choices=['nas-bench-201', 'darts'], help='which nas search space to use')
parser.add_argument('--dataset', default='cifar100', type=str, choices=['cifar10', 'cifar100', 'ImageNet16-120', 'imagenet-1k'], help='Choose between cifar10/100/ImageNet16-120/imagenet-1k')
parser.add_argument('--seed', default=0, type=int, help='manual seed')
args = parser.parse_args()


##### Basic Settings
precision = 3
# init = 'normal'
# init = 'kaiming_uniform'
init = 'kaiming_normal'


if args.space == "nas-bench-201":
    prune_number = 1
    batch_size = 72
    space = "nas-bench-201"  # different spaces of operator candidates, not structure of supernet
    super_type = "basic"  # type of supernet structure
elif args.space == "darts":
    space = "darts"
    super_type = "nasnet-super"
    if args.dataset == "cifar10":
        prune_number = 3
        batch_size = 14
        # batch_size = 6
    elif args.dataset == "imagenet-1k":
        prune_number = 2
        batch_size = 24


timestamp = "{:}".format(time.strftime('%h-%d-%C_%H-%M-%s', time.gmtime(time.time())))


core_cmd = "CUDA_VISIBLE_DEVICES={gpuid} OMP_NUM_THREADS=4 python ./prune_tenas.py \
--save_dir {save_dir} --max_nodes {max_nodes} \
--dataset {dataset} \
--data_path {data_path} \
--search_space_name {space} \
--super_type {super_type} \
--arch_nas_dataset {TORCH_HOME}/NAS-Bench-201-v1_0-e61699.pth \
--track_running_stats 1 \
--workers 0 --rand_seed {seed} \
--timestamp {timestamp} \
--precision {precision} \
--init {init} \
--repeat 3 \
--batch_size {batch_size} \
--prune_number {prune_number} \
".format(
    gpuid=args.gpu,
    save_dir="./output/prune-{space}/{dataset}".format(space=space, dataset=args.dataset),
    max_nodes=4,
    data_path=data_paths[args.dataset],
    dataset=args.dataset,
    TORCH_HOME=TORCH_HOME,
    space=space,
    super_type=super_type,
    seed=args.seed,
    timestamp=timestamp,
    precision=precision,
    init=init,
    batch_size=batch_size,
    prune_number=prune_number
)

os.system(core_cmd)
