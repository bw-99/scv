from fuxictr.pytorch.torch_utils import seed_everything
import argparse
import autotuner
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ECN_tuner_config_KKBox.yaml',
                        help='The config file for para tuning.')
    parser.add_argument('--tag', type=str, default=None,
                        help='Use the tag to determine which expid to run (e.g. 001 for the first expid).')
    parser.add_argument('--gpu', nargs='+', default=[0, 1, 2, 3, 2, 1, 0],
                        help='The list of gpu indexes, -1 for cpu.')
    parser.add_argument('--fix_seed', type=int, default=0, choices=[0,1], help='The gpu index, -1 for cpu')

    args = vars(parser.parse_args())
    gpu_list = args['gpu']
    expid_tag = args['tag']

    if args["fix_seed"] == 1:
        print("fix seed!"*100)
        seed_everything(seed=2024)

    torch.autograd.set_detect_anomaly(True)

    # generate parameter space combinations
    config_dir = autotuner.enumerate_params(args['config'])
    print(config_dir)
    autotuner.grid_search(config_dir, gpu_list, expid_tag, fix_seed= args["fix_seed"])
