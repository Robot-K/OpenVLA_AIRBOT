CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
torchrun --standalone --nnodes 1 --nproc-per-node 7 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir ./data/ \
  --dataset_name airbot_mix \
  --run_root_dir ./workspace \
  --adapter_tmp_dir ./tmp \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug True \
  --wandb_project OpenVLA \
  --wandb_entity dkr21-tsinghua-university \
  --save_steps 400
