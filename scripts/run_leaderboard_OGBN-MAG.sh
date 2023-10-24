GPU=0
DATASET=mag
METHOD=rphgnn
USE_NRL=True 
TRAIN_STRATEGY=cl
USE_INPUT=True
ALL_FEAT=True 
INPUT_DROP_RATE=0.1
DROP_RATE=0.4
HIDDEN_SIZE=512
SQUASH_K=3
EPOCHS=500
MAX_PATIENCE=50
EMBEDDING_SIZE=512
USE_LABEL=True
EVEN_ODD="all"


mkdir cache

for SEED in $(seq 0 9)
do
echo $SEED
python -u main_rphgnn.py \
    --dataset ${DATASET} \
    --method ${METHOD} \
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
    --output_dir outputs/leaderboard_mag/ \
    --gpus ${GPU} \
    --seed ${SEED} > nohup_leaderboard_mag_${SEED}.out 2>&1 
done
