# coding=utf-8

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics

import shutil
import logging
import sys
import time
import os
import json
import datetime
import shortuuid
from argparse import ArgumentParser

from rphgnn.callbacks import EarlyStoppingCallback, LoggingCallback, TensorBoardCallback
from rphgnn.layers.rphgnn_encoder import RpHGNNEncoder
from rphgnn.losses import kl_loss
from rphgnn.utils.metrics_utils import MRR, NDCG
from rphgnn.utils.random_project_utils import create_func_torch_random_project_create_kernel_sparse, torch_random_project_common, torch_random_project_create_kernel_xavier, torch_random_project_create_kernel_xavier_no_norm
from rphgnn.utils.torch_data_utils import NestedDataLoader
from rphgnn.global_configuration import global_config
from rphgnn.utils.argparse_utils import parse_bool
from rphgnn.utils.random_utils import reset_seed
from rphgnn.configs.default_param_config import load_default_param_config
from rphgnn.datasets.load_data import  load_dgl_data
from rphgnn.utils.nested_data_utils import gather_h_y, nested_gather, nested_map
from rphgnn.layers.rphgnn_pre import rphgnn_propagate_and_collect, rphgnn_propagate_and_collect_label


np.set_printoptions(precision=4, suppress=True)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger()


parser = ArgumentParser()

parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--method", type=str, required=True)
parser.add_argument("--use_nrl", type=parse_bool, required=True)
parser.add_argument("--use_input", type=parse_bool, required=True)
parser.add_argument("--use_label", type=parse_bool, required=True)
parser.add_argument("--even_odd", type=str, required=False, default="all")
parser.add_argument("--use_all_feat", type=parse_bool, required=True)
parser.add_argument("--train_strategy", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--gpus", type=str, required=True)
parser.add_argument("--input_drop_rate", type=float, required=False, default=None)
parser.add_argument("--drop_rate", type=float, required=False, default=None)
parser.add_argument("--hidden_size", type=int, required=False, default=None)
parser.add_argument("--squash_k", type=int, required=False, default=None)
parser.add_argument("--num_epochs", type=int, required=False, default=None)
parser.add_argument("--max_patience", type=int, required=False, default=None)
parser.add_argument("--embedding_size", type=int, required=False, default=None)
parser.add_argument("--rps", type=str, required=False, default="sp_3.0", help="random projection strategies")
parser.add_argument("--seed", type=int, required=True)


# sys.argv += cmd.split()
args = parser.parse_args()

method = args.method 
dataset = args.dataset 
use_all_feat = args.use_all_feat
use_nrl = args.use_nrl
use_label = args.use_label
train_strategy = args.train_strategy
use_input_features = args.use_input
output_dir = args.output_dir
gpu_ids = args.gpus
device = "cuda"
data_loader_device = device
even_odd = args.even_odd
random_projection_strategy = args.rps
seed = args.seed

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
reset_seed(seed)
print("seed = ", seed)


global_config.torch_random_project = torch_random_project_common
if random_projection_strategy.startswith("sp"):
    random_projection_sparsity = float(random_projection_strategy.split("_")[1])
    global_config.torch_random_project_create_kernel = create_func_torch_random_project_create_kernel_sparse(s=random_projection_sparsity)
    print("setting random projection strategy: sparse({} ...)".format(random_projection_sparsity))
elif random_projection_strategy == "gaussian":
    global_config.torch_random_project_create_kernel = torch_random_project_create_kernel_xavier
    print("setting random projection strategy: gaussian ...")

elif random_projection_strategy == "gaussian_no_norm":
    global_config.torch_random_project_create_kernel = torch_random_project_create_kernel_xavier_no_norm
    print("setting random projection strategy: gaussian ...")

else:
    raise ValueError("unknown random projection strategy: {}".format(random_projection_strategy))


pre_device = "cpu"
learning_rate = 3e-3
l2_coef = None
norm = "mean"
squash_strategy = "project_norm_sum"
target_h_dtype = torch.float16

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

running_leaderboard_mag = dataset == "mag" and use_label

# hyper-parameters for ogbn-mag learderboard (lp+cl)
if running_leaderboard_mag:
    scheduler_gamma = 0.99
    num_views = 3
    cl_rate = 0.6
    model_save_dir = "saved_models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_save_path = os.path.join(model_save_dir, "leaderboard_mag_seed_{}.pt".format(seed))
else:
    scheduler_gamma = None
    num_views = 1
    cl_rate = None
    model_save_path = None







arg_dict = {**vars(args)}
arg_dict["date"] = timestamp
del arg_dict["output_dir"]
del arg_dict["gpus"]

args_desc_items = []
for key, value in arg_dict.items():
    args_desc_items.append(key)
    args_desc_items.append(str(value))
args_desc = "_".join(args_desc_items)

uuid = "{}_{}".format(timestamp, shortuuid.uuid())

tmp_output_fname = "{}.json.tmp".format(uuid)
tmp_output_fpath = os.path.join(output_dir, tmp_output_fname)

output_fname = "{}.json".format(uuid)
output_fpath = os.path.join(output_dir, output_fname)


print(output_dir)
print(os.path.exists(output_dir))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(tmp_output_fpath, "a", encoding="utf-8") as f:
    f.write("{}\n".format(json.dumps(arg_dict)))


time_dict = {
    "start": time.time()
}

squash_k, inner_k, conv_filters, num_layers_list, hidden_size, merge_mode, input_drop_rate, drop_rate, \
        use_pretrain_features, random_projection_align, input_random_projection_size, target_feat_random_project_size, add_self_group = load_default_param_config(dataset)


embedding_size = None

if args.embedding_size is not None:
    embedding_size = args.embedding_size
    print("reset embedding_size => {}".format(embedding_size))

with torch.no_grad():
    hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index), \
            batch_size, num_epochs, patience, validation_freq, convert_to_tensor = load_dgl_data(
        dataset,
        use_all_feat=use_all_feat,
        embedding_size=embedding_size,
        use_nrl=use_nrl
    )


if args.input_drop_rate is not None:
    input_drop_rate = args.input_drop_rate
    print("reset input_drop_rate => {}".format(input_drop_rate))
    
if args.drop_rate is not None:
    drop_rate = args.drop_rate
    print("reset drop_rate => {}".format(drop_rate))
    
if args.hidden_size is not None:
    hidden_size = args.hidden_size
    print("reset hidden_size => {}".format(hidden_size))

if args.squash_k is not None:
    squash_k = args.squash_k
    print("reset squash_k => {}".format(squash_k))

if args.num_epochs is not None:
    num_epochs = args.num_epochs
    print("reset num_epochs => {}".format(num_epochs))

if args.max_patience is not None:
    patience = args.max_patience
    print("reset patience => {}".format(patience))

y = hetero_graph.ndata["label"][target_node_type].detach().cpu().numpy()

print("train_rate = {}\tvalid_rate = {}\ttest_rate = {}".format(len(train_index) / len(y), len(valid_index) / len(y), len(test_index) / len(y)))

multi_label = len(y.shape) > 1

if multi_label:
    num_classes = y.shape[-1]
else:
    num_classes = y.max() + 1


stage_output_dict = {
    "last": None
}


print("start pre-computation ...")

log_dir = "logs/{}".format(args_desc)

torch_y = torch.tensor(y).long()

if multi_label:
    torch_y = torch_y.float()

train_mask = np.zeros([len(y)])
train_mask[train_index] = 1.0
torch_train_mask = torch.tensor(train_mask).bool()

if even_odd == "odd":
    squash_k *= 2
    print("odd mode, squash_k =", squash_k)



def create_label_target_h_list_list():
    print("using new train_label_feat")
    train_label_feat = torch.ones([len(y), num_classes]).float() / num_classes
    train_label_feat[train_index] = F.one_hot(torch.tensor(y[train_index]), num_classes).float()
        
    label_target_h_list_list = rphgnn_propagate_and_collect_label(hetero_graph, target_node_type, y, train_label_feat)
    label_target_h_list_list = nested_map(label_target_h_list_list, lambda x: x.to(target_h_dtype).to(pre_device))
    return label_target_h_list_list

if use_label:   
    if dataset != "mag":
        raise Exception("use_label is only supported for mag dataset")
    label_target_h_list_list = create_label_target_h_list_list()
else:
    label_target_h_list_list = [] 


feat_target_h_list_list, target_sorted_keys = rphgnn_propagate_and_collect(hetero_graph, 
                        squash_k, 
                        inner_k, 
                        0.0,
                        target_node_type, 
                        use_input_features=use_input_features, squash_strategy=squash_strategy, 
                        train_label_feat=None, 
                        norm=norm,
                        squash_even_odd=even_odd,
                        collect_even_odd=even_odd,
                        squash_self=False,
                        target_feat_random_project_size=target_feat_random_project_size,
                        add_self_group=add_self_group
                        )  

feat_target_h_list_list = nested_map(feat_target_h_list_list, lambda x: x.to(target_h_dtype).to(pre_device))
target_h_list_list = feat_target_h_list_list + label_target_h_list_list


time_dict["pre_compute"] = time.time()
pre_compute_time = time_dict["pre_compute"] - time_dict["start"]
print("pre_compute time: ", pre_compute_time)


accuracy_metric = torchmetrics.Accuracy("multilabel", num_labels=int(num_classes)) if multi_label else torchmetrics.Accuracy("multiclass" if multi_label else "multiclass", num_classes=int(num_classes)) 
if dataset in ["oag_L1", "oag_venue"]:
    metrics_dict = {
        "accuracy": accuracy_metric,
        "ndcg": NDCG(),
        "mrr": MRR()
    }
else:
    metrics_dict = {
        "accuracy": accuracy_metric,
        "micro_f1": torchmetrics.F1Score(task="multilabel", num_labels=int(num_classes), average="micro") if multi_label else torchmetrics.F1Score(task="multiclass", num_classes=int(num_classes), average="micro"),
        "macro_f1": torchmetrics.F1Score(task="multilabel", num_labels=int(num_classes), average="macro") if multi_label else torchmetrics.F1Score(task="multiclass", num_classes=int(num_classes), average="macro"),
    }
metrics_dict = {metric_name: metric.to(device) for metric_name, metric in metrics_dict.items()}


print("create model ====")
model = RpHGNNEncoder(
    conv_filters, 
    [hidden_size] * num_layers_list[0],
    [hidden_size] * (num_layers_list[2] - 1) + [num_classes],
    merge_mode,
    input_shape=nested_map(target_h_list_list, lambda x: list(x.size())),
    input_drop_rate=input_drop_rate,
    drop_rate=drop_rate,
    activation="prelu",
    output_activation="identity",
    metrics_dict=metrics_dict,
    multi_label=multi_label, 
    loss_func=kl_loss if dataset == "oag_L1" else None,
    learning_rate=learning_rate, 
    scheduler_gamma=scheduler_gamma,
    train_strategy=train_strategy, 
    num_views=num_views, 
    cl_rate=cl_rate

    ).to(device)

print(model)

print("number of params:", sum(p.numel() for p in model.parameters()))
logging_callback = LoggingCallback(tmp_output_fpath, {"pre_compute_time": pre_compute_time})
tensor_board_callback = TensorBoardCallback(
    "logs/{}/{}".format(dataset, timestamp)
)



def train_and_eval():
      
    train_h_list_list, train_y = nested_gather([target_h_list_list, torch_y], train_index)
    valid_h_list_list, valid_y = nested_gather([target_h_list_list, torch_y], valid_index)
    test_h_list_list, test_y = nested_gather([target_h_list_list, torch_y], test_index)


    if train_strategy == "common":
        train_data_loader = NestedDataLoader(
            [train_h_list_list, train_y],
            batch_size=batch_size, shuffle=True, device=data_loader_device
        )

    elif train_strategy == "cl":

        seen_mask = torch.zeros_like(torch_y, dtype=torch.bool)
        seen_mask[train_index] = True
        seen_mask[valid_index] = True
        seen_mask[test_index] = True

        def get_seen(x):
            print("get seen ...")
            with torch.no_grad():
                return nested_map(x, lambda x: x[seen_mask])
        
        train_data_loader = NestedDataLoader(
            [get_seen(target_h_list_list), get_seen(torch_y), get_seen(torch_train_mask)],
            batch_size=batch_size, shuffle=True, device=data_loader_device
        )

    else:
        raise Exception("invalid train strategy: {}".format(train_strategy))



    valid_data_loader =NestedDataLoader(
        [valid_h_list_list, valid_y], 
        batch_size=batch_size, shuffle=False, device=data_loader_device
    )
    test_data_loader = NestedDataLoader(
        [test_h_list_list, test_y], 
        batch_size=batch_size, shuffle=False, device=data_loader_device
    )

    if dataset in ["oag_L1", "oag_venue"]:
        early_stop_strategy = "score"
        early_stop_metric_names = ["ndcg"]
    elif dataset in ["mag"]:
        early_stop_strategy = "score"
        early_stop_metric_names = ["accuracy"]
    elif dataset in ["dblp"]:
        early_stop_strategy = "loss"
        early_stop_metric_names = ["macro_f1", "micro_f1"]
    else:
        early_stop_strategy = "score"
        early_stop_metric_names = ["macro_f1", "micro_f1"]

    print("early_stop_metric_names = {}".format(early_stop_metric_names))

    early_stopping_callback = EarlyStoppingCallback(
        early_stop_strategy, early_stop_metric_names, validation_freq, patience, test_data_loader,
        model_save_path=model_save_path
    )
   

    model.fit(
        train_data=train_data_loader,
        epochs=num_epochs,
        validation_data=valid_data_loader,
        validation_freq=validation_freq,
        callbacks=[early_stopping_callback, logging_callback, tensor_board_callback],
    )


    # For ogbn-mag leaderboard, we also evaluate it via OGB's official evaluator
    if running_leaderboard_mag:
        from ogb.nodeproppred import Evaluator
        evaluator = Evaluator("ogbn-mag")

        print("loading saved model ...")
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        
        
        with torch.no_grad():
            valid_y_pred = model.predict(valid_data_loader).argmax(dim=-1, keepdim=True)
            test_y_pred = model.predict(test_data_loader).argmax(dim=-1, keepdim=True)
            ogb_valid_acc = evaluator.eval({
                'y_true': torch_y[valid_index].unsqueeze(-1),
                'y_pred': valid_y_pred
            })['acc']
            ogb_test_acc = evaluator.eval({
                'y_true': torch_y[test_index].unsqueeze(-1),
                'y_pred': test_y_pred
            })['acc']

        print("Results of OGB Evaluator: valid_acc = {}, test_acc = {}".format(ogb_valid_acc, ogb_test_acc))

train_and_eval()

shutil.move(tmp_output_fpath, output_fpath)
print("move tmp file {} => {}".format(tmp_output_fpath, output_fpath))



