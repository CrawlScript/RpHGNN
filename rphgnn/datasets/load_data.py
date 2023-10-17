# coding=utf-8

import torch
import os
import numpy as np

from rphgnn.utils.graph_utils import add_random_feats, dgl_add_all_reversed_edges, dgl_remove_edges
from .hgb import load_imdb, load_freebase, load_dblp, load_hgb_acm
from tqdm import tqdm
import pickle
from gensim.models import Word2Vec
import time
import dgl
from ogb.nodeproppred import DglNodePropPredDataset



def load_mag(device):
    
    # path = args.use_emb
    home_dir = os.getenv("HOME")
    dataset = DglNodePropPredDataset(
        name="ogbn-mag", root=os.path.join(home_dir, ".ogb", "dataset"))
    g, labels = dataset[0]

    # my
    g = g.to(device)

    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]['paper']
    val_nid = splitted_idx["valid"]['paper']
    test_nid = splitted_idx["test"]['paper']
    features = g.nodes['paper'].data['feat']
    g.nodes["paper"].data["feat"] = features.to(device)


    labels = labels['paper'].to(device).squeeze()
    n_classes = int(labels.max() - labels.min()) + 1
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)


    target_node_type = "paper"
    feature_node_types = [target_node_type]

    return g, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid

def load_dgl_mag(embedding_size):
    device = "cpu"
    
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_mag(device)

    g.nodes[target_node_type].data["label"] = labels


    # embedding_size = g.ndata["feat"][target_node_type].size(-1) * 4  
    g = add_random_feats(g, embedding_size, excluded_ntypes=feature_node_types)

    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)

def load_dgl_hgb(dataset, use_all_feat=False, embedding_size=None, random_state=None):

    if dataset == "imdb":
        load_func = load_imdb
    elif dataset == "dblp":
        load_func = load_dblp
    elif dataset == "hgb_acm":
        load_func = load_hgb_acm
    elif dataset == "freebase":
        load_func = load_freebase
    else:
        raise RuntimeError(f"Unsupported dataset {dataset}")

    # dgl_graph, target_node_type, feature_node_types, features, features_dict, labels, num_classes, train_indices, valid_indices, test_indices, train_mask, valid_mask, test_mask = load_func(random_state=random_state)

    dgl_graph, target_node_type, feature_node_types, features, features_dict, labels, _, train_indices, valid_indices, test_indices, _, _, _ = load_func(random_state=random_state)


    if use_all_feat:
        print("use all features ...")
        for int_ntype, value in features_dict.items():
            ntype = str(int_ntype)
            if value is None:
                print("skip None ntype: ", ntype)
            else:
                
                print("set feature for ntype: ", ntype, dgl_graph.num_nodes(ntype), value.shape)
                dgl_graph.nodes[ntype].data["feat"] = torch.tensor(value).to(torch.float32)

        if embedding_size is None:
                embedding_size = features.size(-1)

        dgl_graph = add_random_feats(dgl_graph, embedding_size, 
            excluded_ntypes=[ntype for ntype in dgl_graph.ntypes if "feat" in dgl_graph.nodes[ntype].data]
        )

    else:
        if len(feature_node_types) == 0:
            dgl_graph = add_random_feats(dgl_graph, embedding_size, excluded_ntypes=None)
        else:
            dgl_graph.nodes[target_node_type].data["feat"] = features
            if embedding_size is None:
                embedding_size = features.size(-1)

            dgl_graph = add_random_feats(dgl_graph, embedding_size, 
                excluded_ntypes=[ntype for ntype in dgl_graph.ntypes if "feat" in dgl_graph.nodes[ntype].data]
            )
        
    dgl_graph.nodes[target_node_type].data["label"] = labels

    return dgl_graph, target_node_type, feature_node_types, (train_indices, valid_indices, test_indices)

def load_dgl_hgb_acm(use_all_feat=False, embedding_size=None, random_state=None):
    return load_dgl_hgb("hgb_acm", use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)

def load_dgl_imdb(use_all_feat=False, embedding_size=None, random_state=None):
    return load_dgl_hgb("imdb", use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)

def load_dgl_dblp(use_all_feat=False, embedding_size=None, random_state=None):
    return load_dgl_hgb("dblp", use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)

def load_dgl_freebase(use_all_feat=False, embedding_size=None, random_state=None):
    return load_dgl_hgb("freebase", use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)

def load_oag(device, dataset, data_path="datasets/nars_academic_oag"):
    import pickle
    # assert args.data_dir is not None


    if dataset == "oag_L1":
        graph_file = "graph_L1.pk"
        predict_venue = False
    elif dataset == "oag_venue":
        graph_file = "graph_venue.pk"
        predict_venue = True
    else:
        raise RuntimeError(f"Unsupported dataset {dataset}")
    with open(os.path.join(data_path, graph_file), "rb") as f:
        dataset = pickle.load(f)
    n_classes = dataset["n_classes"]
    graph = dgl.heterograph(dataset["edges"])
    graph = graph.to(device)
    train_nid, val_nid, test_nid = dataset["split"]


    with open(os.path.join(data_path, "paper.npy"), "rb") as f:
        # loading lang features of paper provided by HGT author
        paper_feat = torch.from_numpy(np.load(f)).float().to(device)
    graph.nodes["paper"].data["feat"] = paper_feat[:graph.number_of_nodes("paper")]

    if predict_venue:
        labels = torch.from_numpy(dataset["labels"])
    else:
        labels = torch.zeros(graph.number_of_nodes("paper"), n_classes)
        for key in dataset["labels"]:
            labels[key, dataset["labels"][key]] = 1
    train_nid, val_nid, test_nid = np.array(train_nid), np.array(val_nid), np.array(test_nid)

    # return graph, labels, n_classes, train_nid, val_nid, test_nid

    target_node_type = "paper"
    feature_node_types = [target_node_type]

    return graph, target_node_type, feature_node_types, labels, n_classes, train_nid, val_nid, test_nid

def load_dgl_oag(dataset, data_path="datasets/nars_academic_oag", embedding_size=None):
    g, target_node_type, feature_node_types, labels, n_classes, train_index, valid_index, test_index = load_oag(device="cpu", dataset=dataset, data_path=data_path)

    target_node_type = "paper"

    g = add_random_feats(g, embedding_size, excluded_ntypes=[target_node_type])
    
    g.nodes[target_node_type].data["label"] = labels


    return g, target_node_type, feature_node_types, (train_index, valid_index, test_index)
    # return dgl_graph, target_node_type, (train_index, valid_index, test_index)
    
def nrl_update_features(dataset, hetero_graph, excluded_ntypes, 
                        nrl_pretrain_epochs=40, embedding_size=512):
 
    start_time = time.time()
    nrl_cache_path = os.path.join("./cache/{}.p".format(dataset))

    if os.path.exists(nrl_cache_path):
        print("loading cache: {}".format(nrl_cache_path))
        with open(nrl_cache_path, "rb") as f:
            nrl_embedding_dict = pickle.load(f)
    else:
        
        vocab_corpus = []
        for ntype in hetero_graph.ntypes:
            for i in tqdm(range(hetero_graph.num_nodes(ntype))):
                vocab_corpus.append(["{}_{}".format(ntype, i)])

        
        corpus = []
        for etype in hetero_graph.canonical_etypes:
            if etype[1].startswith("r."):
                print("skip etype: ", etype)
                continue
            row, col = hetero_graph.edges(etype=etype)
            for i, j in tqdm(zip(row, col)):
                corpus.append(["{}_{}".format(etype[0], i), "{}_{}".format(etype[2], j)])

        print("start training word2vec")
        # word2vec_model = Word2Vec(sentences=vocab_corpus, vector_size=embedding_size, window=2, min_count=0, workers=4)
        word2vec_model = Word2Vec(sentences=vocab_corpus, vector_size=embedding_size, window=2, min_count=0, workers=4)
        for i in tqdm(range(nrl_pretrain_epochs)):
            print("train word2vec epoch {}".format(i))
            word2vec_model.train(corpus, total_examples=len(corpus), epochs=1)

        # word2vec_model = Word2Vec(sentences=vocab_corpus, vector_size=embedding_size, window=2, min_count=0, workers=4, negative=20)
        
        # print("train word2vec ...")
        # word2vec_model.train(corpus, total_examples=len(corpus), epochs=nrl_pretrain_epochs)

        nrl_embedding_dict = {}
        for ntype in hetero_graph.ntypes:
            embeddings = np.array([word2vec_model.wv["{}_{}".format(ntype, i)] for i in range(hetero_graph.num_nodes(ntype))])
            nrl_embedding_dict[ntype] = embeddings

        print("saving cache: {}".format(nrl_cache_path))
        with open(nrl_cache_path, "wb") as f:
            pickle.dump(nrl_embedding_dict, f, protocol=4)
    


    print("nrl time: ", time.time() - start_time)


    for ntype in list(hetero_graph.ntypes):
        if ntype not in excluded_ntypes:
            print("using NRL embeddings for featureless nodetype: {}".format(ntype))
            # hetero_graph.x_dict[node_type] = nrl_embedding_dict[node_type]
            hetero_graph.nodes[ntype].data["feat"] = torch.tensor(nrl_embedding_dict[ntype])

    return hetero_graph

def load_dgl_data(dataset, use_all_feat=False, embedding_size=None, use_nrl=False, random_state=None):


    batch_size = 10000
    num_epochs = 510
    patience = 30
    validation_freq = 10
    convert_to_tensor = True

    if dataset == "mag":
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_mag(embedding_size=embedding_size)        

        convert_to_tensor = False
        num_epochs = 100
        patience = 10

    elif dataset in ["oag_L1", "oag_venue"]:

        batch_size = 3000
        if embedding_size is None:
            embedding_size = 128 * 2

        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_oag(dataset, embedding_size=embedding_size)

        convert_to_tensor = False

        num_epochs = 200
        patience = 10



    elif dataset == "imdb":
        
        if embedding_size is None:
            embedding_size = 1024

        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_imdb(use_all_feat=use_all_feat, 
            embedding_size=embedding_size, random_state=random_state)

        etypes_to_remove = set()
        for etype in hetero_graph.canonical_etypes:
            etype_ = etype[1]
            items = list(etype_)
            print("items: ", items)
            if items[0] > items[1]:
                etypes_to_remove.add(etype)
                print("remove items: ", items)

        print("etypes_to_remove: ", etypes_to_remove)

        hetero_graph = dgl_remove_edges(hetero_graph, etypes_to_remove)
        print("remaining etypes: ", hetero_graph.canonical_etypes)

        num_epochs = 500
        patience = 200

        validation_freq = 1

    elif dataset == "dblp":

        if embedding_size is None:
            embedding_size = 1024
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_dblp(use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)
        # hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_dblp(embedding_size=256)

        print("raw etypes: ", hetero_graph.canonical_etypes)
        etypes_to_remove = set()
        for etype in hetero_graph.canonical_etypes:
            etype_ = etype[1]
            items = list(etype_)
            print("items: ", items)
            if items[0] > items[1]:
                etypes_to_remove.add(etype)
                print("remove items: ", items)

        print("etypes_to_remove: ", etypes_to_remove)

        hetero_graph = dgl_remove_edges(hetero_graph, etypes_to_remove)

        print("remaining etypes: ", hetero_graph.canonical_etypes)

        # hetero_graph = dgl_add_duplicated_edges(hetero_graph, 3)
        # print("edges update duplication: ", hetero_graph.canonical_etypes)

        num_epochs = 500
        patience = 30
        # validation_freq = 1
        

        # hetero_graph = hetero_graph.add_reversed_edges(inplace=True)

    elif dataset == "hgb_acm":

        if embedding_size is None:
            embedding_size = 512

        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_hgb_acm(use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)
        # hetero_graph = hetero_graph.add_reversed_edges(inplace=True)
        
        num_epochs = 100
        patience = 20

        validation_freq = 1
        batch_size = 1000

        # for etype in hetero_graph.etypes:
        #     print(etype)
        etypes_to_remove = set()
        for etype in hetero_graph.canonical_etypes:
            etype_ = etype[1]
            items = list(etype_)
            print("items: ", items)
            if etype_[0] == "-" or items[0] > items[1]:
                etypes_to_remove.add(etype)
                print("remove items: ", items)

        print("etypes_to_remove: ", etypes_to_remove)

        hetero_graph = dgl_remove_edges(hetero_graph, etypes_to_remove)
        print("remaining etypes: ", hetero_graph.canonical_etypes)
        

    elif dataset == "freebase":
        num_epochs = 200
        patience = 20
        # validation_freq = 1
        # hetero_graph, target_node_type, (train_index, valid_index, test_index) = load_dgl_freebase(embedding_size=128)
        if embedding_size is None:
            embedding_size = 512
        hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index) = load_dgl_freebase(use_all_feat=use_all_feat, embedding_size=embedding_size, random_state=random_state)



        etypes_to_remove = set()
        for etype in hetero_graph.canonical_etypes:
            etype_ = etype[1]
            items = [int(c) for c in list(etype_)]
            print("items: ", items)
            if items[0] > items[1]:
                etypes_to_remove.add(etype)
                print("remove items: ", items)

        print("etypes_to_remove: ", etypes_to_remove)

        hetero_graph = dgl_remove_edges(hetero_graph, etypes_to_remove)

        print("etypes: ", hetero_graph.canonical_etypes)



    # hetero_graph = dgl_add_label_nodes(hetero_graph, target_node_type, train_index)
    hetero_graph = dgl.add_reverse_edges(hetero_graph, ignore_bipartite=True)
    hetero_graph = dgl_add_all_reversed_edges(hetero_graph)




    if use_nrl:

        if dataset == "freebase":
            excluded_ntypes = []
        else:
            excluded_ntypes = [target_node_type]

        hetero_graph = nrl_update_features(dataset, hetero_graph, excluded_ntypes)


    return hetero_graph, target_node_type, feature_node_types, (train_index, valid_index, test_index), \
           batch_size, num_epochs, patience, validation_freq, convert_to_tensor
