# coding=utf-8

import torch
import dgl

import numpy as np
from rphgnn.utils.random_project_utils import *
from itertools import chain
import logging
from rphgnn.global_configuration import global_config

logger = logging.getLogger()




def torch_svd(x):
    u, s, vh = torch.linalg.svd(x)
    
    print("svd: ", x.size(), u.size(), s.size(), vh.size())
    h = u[:, :s.size(0)] * torch.sqrt(s)
    return h



def get_raw_etype(etype):
    etype_ = etype[1]
    if etype_.startswith("r."):
        return get_reversed_etype(etype)
    else:
        return etype

def rphgnn_propagate_then_update(g, current_k, inner_k, input_x_dim_dict, target_node_type, squash_strategy, norm=None, squash_even_odd="all", squash_self=True, collect_even_odd="all", diag_dict=None, train_label_feat=None):

    with g.local_scope():

        # propagate 
        for etype in g.canonical_etypes:
            last_key = "feat"
            for k_ in range(1, inner_k + 1):
                # print("etype: ", etype, "inner_k_: ", k_)
                odd_or_even = "odd" if k_ % 2 == 1 else "even"
                key = (odd_or_even, k_, etype)
                prop_etype = etype if odd_or_even == "odd" else get_reversed_etype(etype)

                # print("prop_etype: ", prop_etype)

                if norm == "mean":

                    g.update_all(
                        dgl.function.copy_u(last_key, "m"),
                        dgl.function.mean("m", key),

                        # message_func,
                        # dgl.function.sum("m", key),
                        
                        etype=prop_etype
                    )
                    last_key = key

                else:
                    sp = torch.tensor(norm[2])
                    dp = torch.tensor(norm[4])
                
                    def message_func(edges):
                        # return {'m': edges.src[last_key]}
                        return {'m': edges.src[last_key] * \
                                torch.pow(edges.src[("deg", get_reversed_etype(prop_etype))].unsqueeze(-1) + 1e-8, sp) * \
                                torch.pow(edges.dst[("deg", prop_etype)].unsqueeze(-1) + 1e-8, dp)}
                    
                    g.update_all(
                        message_func,
                        dgl.function.sum("m", key),
                        etype=prop_etype
                    )
                    last_key = key
                    

        new_x_dict = {}


        for ntype in g.ntypes:
            # print("deal with {} ...".format(ntype))
            
            # sort keys by (etype, k)
            # [(odd, 1, etype0), (even, 2, etype0), (odd, 3, etype0), (even, 4, etype0), 
            # (odd, 1, etype1), (even, 2, etype1), (odd, 3, etype1), (even, 4, etype1)]
            keys = [key for key in g.nodes[ntype].data.keys() 
                    if isinstance(key, tuple) and key[0] in ["even", "odd"]]
            sort_index = sorted(list(range(len(keys))), key=lambda i: (get_raw_etype(keys[i][-1]), keys[i][1]))
            sorted_keys = [keys[i] for i in sort_index]

        
            x = g.ndata["feat"][ntype]

            # collect for each ntype
            h_list = []
            
            for key in sorted_keys:
                # print(key, g.nodes[ntype].data[key].size())
                h = g.nodes[ntype].data[key]

                # label prop for target node type
                if ntype == target_node_type and diag_dict is not None:

                    if key[0] == "even":
                        diag = diag_dict[key[-1]]
                        # diag = np.expand_dims(diag, axis=-1)
                        h = (h - x * diag) / (1.0 - diag + 1e-8)

                        if train_label_feat is not None:
                            zero_mask = (h.sum(dim=-1) == 0.0)
                            h[zero_mask] = torch.ones_like(h[zero_mask]) / h.size(-1) 
                            print("diag zero to mean for: {} {} {}".format(ntype, key, zero_mask.sum()))

                        print("diag====", key)
                        print("remove diag for: {} {}".format(ntype, key))


                h_list.append(h)


        

            # each even_odd_iter covers an odd and an even, such as (1,2) or (3, 4)
            def get_even_odd_iter(data_list, i):
                """
                input: [(odd, 1, etype0), (even, 2, etype0), (odd, 3, etype0), (even, 4, etype0), (odd, 1, etype1), (even, 2, etype1), (odd, 3, etype1), (even, 4, etype1)]
                
                output: odd+even of a given iteration i
                
                For exampe, if i == 0:
                output => [(odd, 1, etype0), (even, 2, etype0), (odd, 1, etype1), (even, 2, etype1)]
                """
                return list(chain(*list(zip(data_list[i * 2::inner_k], data_list[i * 2 + 1::inner_k]))))

            even_odd_iter_h_list_list = []

            for hop in range(inner_k // 2):
                even_odd_iter_h_list = get_even_odd_iter(h_list, hop)
                even_odd_iter_h_list_list.append(even_odd_iter_h_list)
                even_odd_iter_sorted_keys = get_even_odd_iter(sorted_keys, hop)
                # print("hop sorted keys: ", hop_sorted_keys)

            
            even_odd_iter_sorted_keys = [(key[0], key[2]) for key in even_odd_iter_sorted_keys]

            # push into outputs
            if ntype == target_node_type:
                # print("collect outputs for {}".format(ntype))

                target_h_list_list = [[h.detach().cpu().numpy() for h in hop_h_list] 
                                        for hop_h_list in even_odd_iter_h_list_list]
                target_sorted_keys = even_odd_iter_sorted_keys

                if collect_even_odd != "all":
                    target_h_list_list = [[target_h for target_h, key in zip(target_h_list, target_sorted_keys) if key[0] == collect_even_odd] 
                                          for target_h_list in target_h_list_list]
                    target_sorted_keys = [key for key in target_sorted_keys if key[0] == collect_even_odd]
                    

            

            squash_keys = [("self", )] if squash_self else []
            squash_h_list = [x] if squash_self else []

            for h, key in zip(even_odd_iter_h_list_list[0], even_odd_iter_sorted_keys):
                key_even_odd = key[0]

                use_key = None
                if squash_even_odd == "all":
                    use_key = True
                elif squash_even_odd in ["even", "odd"]:
                    use_key = key_even_odd == squash_even_odd
                else:
                    raise ValueError("squash_even_odd must be all, even or odd")
                
                if use_key:
                    squash_keys.append(key)
                    squash_h_list.append(h)

            if squash_strategy == "sum":
                new_x = torch.stack(squash_h_list, dim=0).sum(dim=0)

            elif squash_strategy == "mean":
                new_x = torch.stack(squash_h_list, dim=0).mean(dim=0)

            elif squash_strategy == "norm_sum":
                normed_squash_h_list = [torch_normalize(h) for h in squash_h_list]
                new_x = torch.stack(normed_squash_h_list, dim=0).sum(dim=0)

            elif squash_strategy == "norm_mean":
                normed_squash_h_list = [torch_normalize(h) for h in squash_h_list]
                new_x = torch.stack(normed_squash_h_list, dim=0).mean(dim=0)

            elif squash_strategy == "norm_mean_norm":
                normed_squash_h_list = [torch_normalize(h) for h in squash_h_list]
                h = torch.stack(normed_squash_h_list, dim=0).mean(dim=0)
                h = torch_normalize(h)
                new_x = h

            elif squash_strategy == "project_norm_sum":
                new_x = torch_random_project_then_sum(
                    squash_h_list,
                    input_x_dim_dict[ntype],
                    norm=True
                )
            
            elif squash_strategy == "project_norm_mean":
                new_x = torch_random_project_then_mean(
                    squash_h_list,
                    input_x_dim_dict[ntype],
                    norm=True
                )
            else:
                raise ValueError("wrong squash_strategy: {}".format(squash_strategy))

            new_x_dict[ntype] = new_x

    # print("update ndata")
    for ntype, new_x in new_x_dict.items():
        g.nodes[ntype].data["feat"] = new_x

    if target_node_type is None:
        target_sorted_keys = None
        target_h_list_list = None

    return (target_h_list_list, target_sorted_keys), g


def compute_deg_dict(g):

    with torch.no_grad():

        deg_dict = {}
        def message_func(edges):
            return {'m': torch.ones([len(edges)])}

        for etype in g.canonical_etypes:
            key = ("deg", etype)
            g.update_all(
                message_func,
                dgl.function.sum("m", key),
                etype=etype
            )
            deg = g.ndata[key][etype[-1]]
            deg_dict[etype] = deg

    return deg_dict


def compute_diag_dict(g):
    import scipy.sparse as sp
    import numpy as np

    diag_dict = {}

    def norm_adj(adj):
        deg = np.array(adj.sum(axis=-1)).flatten()
        inv_deg = 1.0 / deg
        inv_deg[np.isnan(inv_deg)] = 0.0
        inv_deg[np.isinf(inv_deg)] = 0.0

        normed_adj = sp.diags(inv_deg) @ adj
        return normed_adj

    with torch.no_grad():

        for etype in g.canonical_etypes:
            src, dst = g.edges(etype=etype)
            src = src.detach().cpu().numpy()
            dst = dst.detach().cpu().numpy()

            shape = [g.num_nodes(etype[0]), g.num_nodes(etype[-1])]

            adj = sp.csr_matrix((np.ones_like(src), (src, dst)), shape=shape)

            
            diag = (norm_adj(adj).multiply(norm_adj(adj.T).T)).sum(axis=-1)
            diag = np.array(diag).flatten().astype(np.float32)
        
            diag = np.expand_dims(diag, axis=-1)
            diag_dict[etype] = torch.tensor(diag)

            # print("compute diag for {}: {}".format(etype, diag))


    return diag_dict




def rphgnn_propagate_and_collect(g, k, inner_k, alpha, target_node_type, use_input_features, squash_strategy, train_label_feat, norm, squash_even_odd, collect_even_odd, squash_self=False, target_feat_random_project_size=None, add_self_group=False):

    with torch.no_grad():

        raw_input_target_x = g.ndata["feat"][target_node_type]

        with g.local_scope():

            featureless_node_types = [ntype for ntype in g.ntypes if ntype != target_node_type]
            embedding_size = g.ndata["feat"][featureless_node_types[0]].size(-1)

            if target_feat_random_project_size is not None:
                new_x = global_config.torch_random_project(raw_input_target_x, target_feat_random_project_size, norm=True)
                g.nodes[target_node_type].data["feat"] = new_x
                print("random_project_target_feat {} => {}...".format(raw_input_target_x.size(-1), new_x.size(-1)))

            if train_label_feat is not None:

                num_classes = train_label_feat.size(-1)
                for ntype in g.ntypes:
                    if ntype == target_node_type:
                        g.nodes[ntype].data["feat"] = train_label_feat
                    else:
                        g.nodes[ntype].data["feat"] = torch.ones([g.num_nodes(ntype), num_classes]) / num_classes

                diag_dict = compute_diag_dict(g)

            else:
                diag_dict = None

            input_x_dim_dict = {
                ntype: g.ndata["feat"][ntype].size(-1)
                for ntype in g.ntypes
            }

            input_x_dict = {
                ntype: g.ndata["feat"][ntype] for ntype in g.ntypes
            }

                

            input_target_x = g.ndata["feat"][target_node_type]#.detach().cpu().numpy()
            target_h_list_list = []
            for k_ in range(k):
                # print("start propagate {} ...".format(k_))
                
                print("start {} propagate-then-update iteration {} ...".format("feat" if train_label_feat is None else "pre-label", k_))
                (target_h_list_list_, target_sorted_keys), g = rphgnn_propagate_then_update(g, k_, inner_k, input_x_dim_dict, target_node_type, squash_strategy=squash_strategy, norm=norm, squash_even_odd=squash_even_odd, collect_even_odd=collect_even_odd, squash_self=squash_self, diag_dict=diag_dict, train_label_feat=train_label_feat)
               
                target_h_list_list.extend(target_h_list_list_)


                for ntype in g.ntypes:
                    g.nodes[ntype].data["feat"] = g.nodes[ntype].data["feat"] * (1 - alpha) + input_x_dict[ntype] * alpha


            target_h_list_list = [list(target_h_list) for target_h_list in zip(*target_h_list_list)]


        target_sorted_keys_ = target_sorted_keys[:]
        target_h_list_list_ = target_h_list_list[:]


        if train_label_feat is not None:

            target_sorted_keys = []
            target_h_list_list = []
            for key, target_h_list in zip(target_sorted_keys_, target_h_list_list_):
                if key[0] == "even":
                    target_sorted_keys.append(key)
                    target_h_list_list.append(target_h_list)
                elif key[0] == "odd":
                    etype = key[-1]
                    if etype[0] == etype[-1]:
                        print("add homo for label: ", key)
                        target_sorted_keys.append(key)
                        target_h_list_list.append(target_h_list)

            target_h_list_list = [target_h_list[-1:] for target_h_list in target_h_list_list]

            
        if use_input_features:
            for target_h_list, key in zip(target_h_list_list, target_sorted_keys):
                if key[0] in ["even", "self"]:

                    print("add input x to {}".format(key))
                    x = input_target_x
                    x = x.detach().cpu().numpy()

                    target_h_list.insert(0, x)


        # for target_h_list in target_h_list_list:
        #     print("context: ")
        #     for target_h in target_h_list:
        #         print(target_h.shape)

    if add_self_group:
        target_h_list_list.append([raw_input_target_x.detach().cpu().numpy()])
        target_sorted_keys.append(("self",))

    print("target_sorted_keys: ", target_sorted_keys)
    target_h_list_list = [np.stack(target_h_list, axis=1) for target_h_list in target_h_list_list]
    return target_h_list_list, target_sorted_keys



def rphgnn_propagate_and_collect_label(hetero_graph, target_node_type, y, train_label_feat):

    label_target_h_list_list, _ = rphgnn_propagate_and_collect(hetero_graph, 
                1, 
                2, 
                0.0,
                target_node_type, use_input_features=False, 
                squash_strategy="mean", 
                train_label_feat=train_label_feat, 
                norm="mean",  
                squash_even_odd="all",
                collect_even_odd="all"
                )  
    

    return label_target_h_list_list