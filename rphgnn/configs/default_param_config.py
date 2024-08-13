# coding=utf-8

def load_default_param_config(dataset):

    use_pretrain_features = False

    random_projection_align = False
    input_random_projection_size = None

    merge_mode = "concat"
    target_feat_random_project_size = None
    add_self_group = False

    if dataset == "mag":

        input_drop_rate = 0.1
        drop_rate = 0.4

        hidden_size = 512

        inner_k = 2
        squash_k = 3

        
        conv_filters = 2
        num_layers_list = [2, 0, 2]


    elif dataset == "oag_venue":
        
        input_drop_rate = 0.5
        drop_rate = 0.5
        hidden_size = 512
        
        inner_k = 2
        squash_k = 3

        conv_filters = 2
        num_layers_list = [2, 0, 2]

        merge_mode = "mean"

        target_feat_random_project_size = 256
        add_self_group = True

    elif dataset == "oag_L1":
        
        input_drop_rate = 0.5
        drop_rate = 0.5
        hidden_size = 512
        
        inner_k = 2
        squash_k = 3

        conv_filters = 2
        num_layers_list = [2, 0, 2]

        merge_mode = "mean"
        target_feat_random_project_size = 256
        add_self_group = True


    elif dataset == "imdb":

        
        input_drop_rate = 0.8
        drop_rate = 0.8

        hidden_size = 512

        inner_k = 2
        squash_k = 4
            
        conv_filters = 2
        num_layers_list = [2, 0, 2]

    elif dataset == "dblp":


        input_drop_rate = 0.8
        drop_rate = 0.7

        input_random_projection_size = None


        hidden_size = 256

        inner_k = 2
        
        squash_k = 5


        conv_filters = 2
        num_layers_list = [2, 0, 2]


    elif dataset == "hgb_acm":



        input_drop_rate = 0.7
        drop_rate = 0.7

        input_random_projection_size = None
        
        hidden_size = 64
        
        
        inner_k = 2
        
        squash_k = 1


        conv_filters = 2
        num_layers_list = [2, 0, 2]
        merge_mode = "mean"


    elif dataset == "freebase":


        input_drop_rate = 0.7
        drop_rate = 0.7
        hidden_size = 128
        
        inner_k = 2
        
        squash_k = 5

        # k = 3
        validation_freq = 10
        conv_filters = 2
        num_layers_list = [1, 0, 1]
        

    return squash_k, inner_k, conv_filters, num_layers_list, hidden_size, merge_mode, input_drop_rate, drop_rate, \
           use_pretrain_features, random_projection_align, input_random_projection_size, target_feat_random_project_size, add_self_group




