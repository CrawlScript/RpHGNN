# RpHGNN
Source code and dataset of the paper "Source code and dataset of the paper [Efficient Heterogeneous Graph Learning via Random Projection](https://arxiv.org/abs/2310.14481)



## Homepage and Paper

+ Homepage (RpHGNN): [https://github.com/CrawlScript/RpHGNN](https://github.com/CrawlScript/RpHGNN)
+ Paper: [Efficient Heterogeneous Graph Learning via Random Projection ](https://arxiv.org/abs/2310.14481) 




## Requirements

+ Linux
+ Python 3.7
+ torch==1.12.1+cu113
+ torchmetrics==0.11.4
+ dgl==1.0.2+cu113
+ ogb==1.3.5
+ shortuuid==1.0.11
+ pandas==1.3.5
+ gensim==4.2.0
+ numpy==1.21.6
+ tqdm==4.64.1


## Download Preparation

For HGB datasets (ACM, DBLP, Freebase, and IMDB)：

```shell
sh download_hgb_datasets.sh 
```

For OAG-Venue and OAG-L1-Field, we follow NARS' data prepatation in [https://github.com/facebookresearch/NARS/tree/main/oag_dataset](https://github.com/facebookresearch/NARS/tree/main/oag_dataset).
After generating *.pk and *.npy files, you have to:
- put these files in the directory 
- rename graph_field.pk to graph_L1.pk


For OGBN-MAG, the code will automatically download it via the ogb package.


For OAG-Venue and OAG-L1-Field, we adhere to NARS' data preparation instructions found at [https://github.com/facebookresearch/NARS/tree/main/oag_dataset](https://github.com/facebookresearch/NARS/tree/main/oag_dataset).
After generating *.pk and *.npy files, you should:
- Place these files in the directory `./datasets/nars_academic_oag/`.
- Rename graph_field.pk to graph_L1.pk.



## Run RpHGNN

You can run MGDCF with the following command:
```shell
sh scripts/run_ACM.sh

sh scripts/run_DBLP.sh

sh scripts/run_Freebase.sh

sh scripts/run_IMDB.sh

sh scripts/run_OGBN-MAG.sh

sh scripts/run_OAG-Venue.sh

sh scripts/run_OAG-L1-Field.sh
```


## Run RpHGNN for OGB Leaderboards (ogbn-mag)

To reproduce the results on the OGB Leaderboards (ogbn-mag), follow the steps below:

- Preparing Pre-trained Embeddings (Optional):
    - If the cache/mag.p file does not exist (embeddings pre-trained via LINE [1]), our code will automatically pre-train it and save the pre-trained embeddings in the specified path.
    - Alternatively, if you'd prefer to skip the pre-training step, download the pre-trained embeddings mag.p directly from [Google Drive](https://drive.google.com/file/d/1Q7gD1xpmLeFJu5xWWY3nwa46cM8xYClH/view?usp=sharing) and place it in the `cache` directory.


- Execute the script:

    ```shell
    sh scripts/run_leaderboard_OGBN-MAG.sh
    ```
    
    This script will run the training and evaluation using random seeds from 0 to 9. The output for seed i will be saved in the file nohup_leaderboard_mag_i.out.


References:
- [1] Jian Tang, Meng Qu, Mingzhe Wang, Ming Zhang, Jun Yan, and Qiaozhu Mei. "Line: Large-scale information network embedding." In Proceedings of the 24th international conference on world wide web, pp. 1067-1077. 2015.



## Cite

If you use MGDCF in a scientific publication, we would appreciate citations to the following paper:

```
@misc{hu2023efficient,
      title={Efficient Heterogeneous Graph Learning via Random Projection}, 
      author={Jun Hu and Bryan Hooi and Bingsheng He},
      year={2023},
      eprint={2310.14481},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


