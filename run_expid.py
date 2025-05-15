# =========================================================================
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

import sys
sys.path.append("./")
sys.path.append("/home/lhh/code")
import os
os.chdir(os.path.dirname(os.path.realpath(__file__)))
import sys
import logging
from datetime import datetime
from fuxictr.utils import load_config, set_logger, print_to_json, print_to_list
from fuxictr.features import FeatureMap
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.pytorch.dataloaders import RankDataLoader
from fuxictr.preprocess import build_dataset
from custom_fp import CustomizedFeatureProcessor as FeatureProcessor 
import torch
import src as model_zoo
import gc
import argparse
import os
from pathlib import Path
import os
import subprocess
import numpy as np
import torch.nn as nn

def block_frobenius_norm(mat, num_features, block_size=16):
    norm_mat = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(num_features):
            block = mat[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
            norm_mat[i, j] = np.linalg.norm(block, ord='fro')
    return norm_mat

def get_model(config_path, experiment_id, **kwargs):
    params = load_config(config_path, experiment_id)
    params.update(kwargs)
    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))

    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model
    return model


if __name__ == '__main__':
    ''' Usage: python run_expid.py --config {config_dir} --expid {experiment_id} --gpu {gpu_device_id}
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/', help='The config directory.')
    parser.add_argument('--expid', type=str, default='DCNv3_Criteo', help='The experiment id to run.')
    parser.add_argument('--gpu', type=int, default=0 , help='The gpu index, -1 for cpu')
    parser.add_argument('--remove_model', type=int, default=1 , help='The gpu index, -1 for cpu')
    parser.add_argument('--mask_rate', type=float, default=0, help='The gpu index, -1 for cpu')
    parser.add_argument('--save_feature_interaction', type=int, default=0, choices=[0,1], help='The gpu index, -1 for cpu')
    parser.add_argument('--fix_seed', type=int, default=0, choices=[0,1], help='The gpu index, -1 for cpu')

    
    args = vars(parser.parse_args())
    
    experiment_id = args['expid']
    mask_rate = args['mask_rate']
    print("remove_model"*10 , args["remove_model"] == 1)
    
    params = load_config(args['config'], experiment_id)
    # params["epochs"] = 1
    print('**************')
    print(args['config'], experiment_id)
    params['gpu'] = args['gpu']
    params["experiment_id"] = f"{experiment_id}"
    params["mask_rate"] = args["mask_rate"]
    params["save_feature_interaction"] = args["save_feature_interaction"]
    print('**************')
    print(params["experiment_id"])
    set_logger(params)
    logging.info("Params: " + print_to_json(params))

    if args["fix_seed"] == 1:
        print("fix seed!"*100)
        seed_everything(seed=params['seed'])


    data_dir = os.path.join(params['data_root'], params['dataset_id'])
    print('*******************')
    print(data_dir)
    feature_map_json = os.path.join(data_dir, "feature_map.json")
    if params["data_format"] == "csv":
        feature_encoder = FeatureProcessor(**params)
        params["train_data"], params["valid_data"], params["test_data"] = \
            build_dataset(feature_encoder, **params)
    feature_map = FeatureMap(params['dataset_id'], data_dir)
    feature_map.load(feature_map_json, params)
    logging.info("Feature specs: " + print_to_json(feature_map.features))
    # LogCNv3_exp_var_iPinYou_x1_058_febf35a5

    model_class = getattr(model_zoo, params['model'])
    model = model_class(feature_map, **params)
    model.count_parameters() # print number of parameters used in model

    train_gen, valid_gen = RankDataLoader(feature_map, stage='train', **params).make_iterator()
    model.fit(train_gen, validation_data=valid_gen, **params)

    logging.info('****** Validation evaluation ******')
    valid_result = model.evaluate(valid_gen)
    del train_gen, valid_gen
    gc.collect()
    
    logging.info('******** Test evaluation ********')
    test_gen = RankDataLoader(feature_map, stage='test', **params).make_iterator()
    test_result = {}
    if test_gen:
      test_result = model.evaluate(test_gen)
    
    result_filename = Path(args['config']).name.replace(".yaml", "") + '.csv'
    with open(result_filename, 'a+') as fw:
        fw.write(' {},[command] python {},[exp_id] {},[dataset_id] {},[train] {},[val] {},[test] {}\n' \
            .format(datetime.now().strftime('%Y%m%d-%H%M%S'), 
                    ' '.join(sys.argv), experiment_id, params['dataset_id'],
                    "N.A.", print_to_list(valid_result), print_to_list(test_result)))

    model_dir = os.path.join(params["model_root"], feature_map.dataset_id)

    # torch.save(test_result, f"{experiment_id}_{mask_rate}_plot.pt")

    # if "DCNv2" in experiment_id:
    #     adj_matrix = torch.load(f"./{experiment_id}_weight.pt")
    #     embedding_dim = params["embedding_dim"]

    #     weight_mat = adj_matrix.detach().cpu().numpy()  # torch tensor -> numpy
    #     block_size = embedding_dim
    #     num_features = feature_map.num_fields
    #     norm_mat = block_frobenius_norm(weight_mat,num_features=num_features, block_size=block_size)

    #     th_lst, result_lst = [], []
    #     for th in np.arange(0, 1.1, 0.1):
    #         tmp_matrix = torch.from_numpy(norm_mat.copy())
    #         flat = tmp_matrix.flatten()
    #         _, indices = torch.topk(flat.abs(), max(1, min(tmp_matrix.numel(), int(tmp_matrix.numel() * th))), largest=False)
    #         mask = torch.ones_like(flat)
    #         mask[indices] = 0
    #         binary_mat = mask.view(tmp_matrix.shape[0], tmp_matrix.shape[1])

    #         expanded_mat = torch.zeros((norm_mat.shape[0] * block_size, norm_mat.shape[1] * block_size), dtype=int)

    #         for i in range(norm_mat.shape[0]):
    #             for j in range(norm_mat.shape[1]):
    #                 if binary_mat[i, j] == 1:
    #                     expanded_mat[i*block_size:(i+1)*block_size,
    #                                 j*block_size:(j+1)*block_size] = 1
    #         expanded_mat = expanded_mat.to(device=model.device)
    #         model.crossnet.mask = nn.Parameter(expanded_mat, requires_grad=False)
    #         test_result = model.evaluate(test_gen)
    #         print(th, test_result, model.crossnet.mask.sum())
    #         del model.crossnet.mask
    #         gc.collect()
    #         torch.cuda.empty_cache()

    #         th_lst.append(th)
    #         result_lst.append(test_result)
    #     torch.save({"th_lst": th_lst, "result_lst": result_lst}, f"{experiment_id}_test_plot.pt")

    # elif "FinalNet" in experiment_id:
    #     th_lst, result_lst = [], []
    #     for th in np.arange(0, 1.1, 0.1):
    #         block1_mask = torch.load(f"./{experiment_id}_weight.pt")
    #         flat = block1_mask.view(-1)
    #         _, indices = torch.topk(flat.abs(), max(1, min(flat.numel(), int(block1_mask.numel() * th))), largest=False)
    #         mask = torch.ones_like(flat)
    #         mask[indices] = 0
    #         binary_mat1 = mask.view(block1_mask.shape[0], block1_mask.shape[1]).to(device=model.device)
    #         mask = nn.Parameter(binary_mat1, requires_grad=False)
    #         model.block1.layer[0].mask = mask

    #         test_result = model.evaluate(test_gen)
    #         print(th, test_result, model.block1.layer[0].mask.sum())
    #         th_lst.append(th)
    #         result_lst.append(test_result)
    #     torch.save({"th_lst": th_lst, "result_lst": result_lst}, f"{experiment_id}_test_plot.pt")
    
    # elif "EulerNet" in experiment_id:
    #     th_lst, result_lst = [], []
    #     for th in np.arange(0, 1.1, 0.1):
    #         inter_orders = torch.load(f"./{experiment_id}_weight.pt")
    #         flat = inter_orders.view(-1)
    #         _, indices = torch.topk(flat.abs(), max(1, min(inter_orders.numel(), int(inter_orders.numel() * th))), largest=False)
    #         mask = torch.ones_like(flat)
    #         mask[indices] = 0
    #         binary_mat = mask.view(inter_orders.shape[0], inter_orders.shape[1]).to(device=model.device)
    #         mask = nn.Parameter(binary_mat, requires_grad=False)
    #         model.Euler_interaction_layers[0].mask = mask

    #         test_result = model.evaluate(test_gen)
    #         print(th, test_result, model.Euler_interaction_layers[0].mask.sum())
    #         th_lst.append(th)
    #         result_lst.append(test_result)
    #     torch.save({"th_lst": th_lst, "result_lst": result_lst}, f"{experiment_id}_test_plot.pt")

    # elif "AdaGIN" in experiment_id:
    #     th_lst, result_lst = [], []
    #     for th in np.arange(0, 1.1, 0.1):
    #         inter_orders = torch.load(f"./{experiment_id}_weight.pt")
    #         flat = inter_orders.view(-1)
    #         _, indices = torch.topk(flat.abs(), max(1, min(inter_orders.numel(), int(inter_orders.numel() * th))), largest=False)
    #         mask = torch.ones_like(flat)
    #         mask[indices] = 0
    #         binary_mat = mask.view(inter_orders.shape[0], inter_orders.shape[1]).to(device=model.device)
    #         mask = nn.Parameter(binary_mat, requires_grad=False)
    #         model.AutoGraph.mask = mask

    #         test_result = model.evaluate(test_gen)
    #         print(th, test_result, model.AutoGraph.mask.sum())
    #         th_lst.append(th)
    #         result_lst.append(test_result)
    #     torch.save({"th_lst": th_lst, "result_lst": result_lst}, f"{experiment_id}_test_plot.pt")
    
    # elif "AutoInt" in experiment_id:
    #     th_lst, result_lst = [], []
    #     for th in np.arange(0, 1.1, 0.1):
    #         inter_orders = torch.load(f"./{experiment_id}_weight.pt")  # shape: [num_heads, num_feat, num_feat]
    #         num_heads, num_feat, _ = inter_orders.shape
    #         binary_mat = torch.ones_like(inter_orders)

    #         for h in range(num_heads):
    #             flat = inter_orders[h].flatten()
    #             k = max(1, int(flat.numel() * th))
    #             _, indices = torch.topk(flat.abs(), k, largest=False)
    #             mask = torch.ones_like(flat)
    #             mask[indices] = 0
    #             binary_mat[h] = mask.view(num_feat, num_feat)
    #         binary_mat = 1 - binary_mat
    #         binary_mat = binary_mat.unsqueeze(0)
    #         mask = nn.Parameter(binary_mat.to(device=model.device), requires_grad=False)
    #         model.self_attention[0].mask = mask

    #         test_result = model.evaluate(test_gen)
    #         print(th, test_result, model.self_attention[0].mask.sum())
    #         th_lst.append(th)
    #         result_lst.append(test_result)
    #     torch.save({"th_lst": th_lst, "result_lst": result_lst}, f"{experiment_id}_test_plot.pt")


    if args["remove_model"] == 1:
        logging.info('******** Remove Model Weights ********')
        subprocess.run(f"rm -rf ./checkpoints/*/{experiment_id}.model", shell=True, check=True)
