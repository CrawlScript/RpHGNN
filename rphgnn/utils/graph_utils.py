# coding=utf-8

import dgl
import numpy as np
import torch
from rphgnn.global_configuration import global_config

def dgl_remove_edges(g, etypes_to_remove):
    etypes_to_remove = set(etypes_to_remove)

    edge_dict = {}
    for etype in list(g.canonical_etypes):
        if etype not in etypes_to_remove:
            edge_dict[etype] = g.edges(etype=etype)

    new_g = dgl.heterograph(edge_dict)

    for key in g.ndata:
        print("key = ", key)
        value = {ntype: data for ntype, data in g.ndata[key].items() if ntype in new_g.ntypes}
        new_g.ndata[key] = value

    return new_g

def dgl_add_all_reversed_edges(g):
    edge_dict = {}
    for etype in list(g.canonical_etypes):
        col, row = g.edges(etype=etype)
        edge_dict[etype] = (col, row)

        if etype[0] != etype[2]:
            new_etype = (etype[2], "r.{}".format(etype[1]), etype[0])
            edge_dict[new_etype] = (row, col)

    new_g = dgl.heterograph(edge_dict)

    for key in g.ndata:
        print("key = ", key)
        new_g.ndata[key] = g.ndata[key]

    return new_g


def add_random_feats(hetero_graph, embedding_size, excluded_ntypes=None):
    def normalize(x):
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def create_embedding_for_node_type(ntype):
        num_nodes = hetero_graph.num_nodes(ntype)
        print("start random feature")
        embeddings = torch.randn([num_nodes, embedding_size], generator=global_config.embedding_generator) / np.sqrt(embedding_size)
        return embeddings
        
    for ntype in list(hetero_graph.ntypes):
        if excluded_ntypes is None or ntype not in excluded_ntypes:
            print("set data: ", ntype)
            hetero_graph.nodes[ntype].data["feat"] = create_embedding_for_node_type(ntype)

    return hetero_graph
