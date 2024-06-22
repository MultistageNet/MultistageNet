# chmod +x job.sh
# ./job.sh

#!/bin/sh


#cd ..

python3 main_multiple_seed.py \
    --seq_len 12 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --n_epoch 100 \
    --d_model 64 \
    --d_ff 256 \
    --num_heads 8 \
    --kernel_size 3 \
    --num_causal_layers 3 \
    --num_mmp_layers 3 \
    --n_regressor_layer 3 \
    --dropout_p 0.1 \
    --activation GELU \
    --device_num 0

