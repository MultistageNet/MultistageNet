
python3 main.py \
    --dataset LiquidSugar \
    --target yield_flow \
    --task_type forecasting \
    --collection_interval 15 \
    --n_stage 3 \
    --seq_len 12 \
    --pred_len 1 \
    --batch_size 64 \
    --num_workers 0 \
    --learning_rate 0.001 \
    --n_epoch 100 \
    --model_name MultistageNet \
    --d_model 32 \
    --d_ff 128 \
    --kernel_size 3 \
    --n_attention_heads 2 \
    --n_temporal_layers 3 \
    --n_multistage_layers 3 \
    --n_regressor_layers 3 \
    --dropout_p 0.2 \
    --activation GELU \
    --seed 0 \
    --device_num 0
    
