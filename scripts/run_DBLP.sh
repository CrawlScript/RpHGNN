SEED=1
GPU=0
DATASET=dblp
METHOD=rphgnn
USE_NRL=False 
TRAIN_STRATEGY=common
USE_INPUT=False
ALL_FEAT=True 
INPUT_DROP_RATE=0.8  
DROP_RATE=0.8 
HIDDEN_SIZE=512
SQUASH_K=5
EPOCHS=500
MAX_PATIENCE=30
EMBEDDING_SIZE=1024
USE_LABEL=False
EVEN_ODD="all"

python -u main_rphgnn.py \
    --method ${METHOD} \
    --dataset ${DATASET} \
    --use_nrl ${USE_NRL} \
    --use_label ${USE_LABEL} \
    --even_odd ${EVEN_ODD} \
    --train_strategy ${TRAIN_STRATEGY} \
    --use_input ${USE_INPUT} \
    --input_drop_rate ${INPUT_DROP_RATE} \
    --drop_rate ${DROP_RATE} \
    --hidden_size ${HIDDEN_SIZE} \
    --squash_k ${SQUASH_K} \
    --num_epochs ${EPOCHS} \
    --max_patience ${MAX_PATIENCE} \
    --embedding_size ${EMBEDDING_SIZE} \
    --use_all_feat ${ALL_FEAT} \
    --output_dir outputs/${DATASET}/${METHOD}/ \
    --seed ${SEED} \
    --gpus ${GPU}  
