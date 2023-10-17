SEED=0
GPU=0
DATASET=oag_L1
METHOD=rphgnn
USE_NRL=False 
TRAIN_STRATEGY=common
USE_INPUT=False
ALL_FEAT=True 
INPUT_DROP_RATE=0.3
DROP_RATE=0.5 
HIDDEN_SIZE=512
SQUASH_K=3
EPOCHS=200
MAX_PATIENCE=0
EMBEDDING_SIZE=384
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
