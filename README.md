# ADePT
This the implementation of [ADePT: Adaptive Decomposed Prompt Tuning for Parameter-Efficient Fine-tuning](https://openreview.net/forum?id=fswihJIYbd)

# The script for bash
```
MODEL=t5-base
MAX_LENGTH=256
MAX_STEPS=30000


PREFIX_LENGTH=60
R=19
for TASK_NAME in rte; do
  for LORA_LR in 1e-4; do
      for lr in 3e-1 4e-1; do
            CUDA_VISIBLE_DEVICES=0 python train.py \
                --peft_type PROMPT_TUNING_LORAX \
                --lora_embedding_lr ${LORA_LR} \
                --learning_rate ${lr} \
                --prefix_length ${PREFIX_LENGTH} \
                --r ${R} \
                --task_name ${TASK_NAME} \
                --dataset_config_name en \
                --model_name_or_path ${MODEL} \
                --do_train \
                --do_eval \
                --do_predict \
                --per_device_train_batch_size 32 \
                --per_device_eval_batch_size 32 \
                --max_seq_length ${MAX_LENGTH} \
                --save_strategy steps \
                --evaluation_strategy steps \
                --max_steps ${MAX_STEPS} \
                --eval_steps 1000 \
                --save_steps 1000 \
                --warmup_steps 500 \
                --weight_decay 1e-2 \
                --load_best_model_at_end \
                --gradient_accumulation_steps 1\
                --save_total_limit 1 \
                --output_dir saved_adept${MODEL}/${TASK_NAME}_lr${lr}_loralr${LORA_LR}_pl${PREFIX_LENGTH}_r${R}_st${MAX_STEPS};
        done;
    done;
done
```

# Acknowledgments
Our code is built upon https://github.com/ZhengxiangShi/DePT

