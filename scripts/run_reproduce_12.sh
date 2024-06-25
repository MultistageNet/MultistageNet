
python3 main_multiple_seed.py \
    --seq_len 12 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --n_epoch 100 \
    --d_model 32 \
    --d_ff 128 \
    --n_attention_heads 2 \
    --kernel_size 3 \
    --n_temporal_layers 3 \
    --n_multistage_layers 3 \
    --n_regressor_layers 3 \
    --dropout_p 0.2 \
    --activation GELU \
    --device_num 0
