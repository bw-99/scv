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
# import sys
# sys.path.append("/mnt/public/lhh/code/")
import sys
from fuxictr.pytorch.torch_utils import seed_everything
from fuxictr.utils import load_dataset_config
from run_expid import train_expid

import gc
import argparse
from functools import partial
from ray import tune
from ray import train
import yaml
import optuna as ot
from ray.tune.search.optuna import OptunaSearch

sys.path.append("./")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/ECN_tuner_config_KKBox.yaml',
                        help='The config file for para tuning.')
    args = vars(parser.parse_args())
    seed_everything(2024)

    model_config = yaml.load(open(args['config'], "r"), Loader=yaml.FullLoader)
    base_expid = model_config['base_expid']
    base_model_config = yaml.load(open("./config/model_config.yaml", "r"), Loader=yaml.FullLoader)[base_expid]

    tuner_space = model_config['tuner_space']
    for k, v in tuner_space.items():
        if type(v) is list:
            tuner_space[k] = v[0] if len(v) == 1 else tune.choice(v)
        else:
            tuner_space[k] = v
            
    data_config = load_dataset_config("./config", model_config['dataset_id'])
    params = {}
    params.update(base_model_config)
    params.update(model_config)
    params.update(data_config)
    params['gpu'] = 0
    print("#"*100)
    print(params)
    print(tuner_space)
    algo = OptunaSearch(
        sampler=ot.samplers.TPESampler(seed=2024),
        metric=["AUC", "logloss"],
        mode=["max", "min"]
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            partial(train_expid, params=params, config_path=args['config']), 
            resources={"cpu": 2, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            search_alg=algo,
            num_samples=100
        ),
        param_space=tuner_space,
        run_config=train.RunConfig(
            stop={"training_iteration": 10},
        ),
    )
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    df = results.get_dataframe()
    df.to_csv(f"{args['config'].split('/')[-1].split('.')[0]}_optuna.csv", index=False)