export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python run.py \
   --output_dir saved_models/r_drop \
   --model_name_or_path microsoft/unixcoder-base-nine \
   --do_train \
   --train_data_file ./storage/dataset/train.json \
   --eval_data_file ./storage/dataset/valid.json \
   --num_train_epochs 20 \
   --block_size 400 \
   --train_batch_size 32 \
   --eval_batch_size 32 \
   --learning_rate 2e-5 \
   --max_grad_norm 1.0 \
   --seed 99 \
   --r_drop