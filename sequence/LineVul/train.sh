export CUDA_VISIBLE_DEVICES=0



nohup python linevul_main.py \
  --output_dir=./storage/saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=storage/dataset/train.json \
  --valid_data_file=storage/dataset/valid.json \
  --test_data_file=storage/dataset/test.json \
  --dataset=vul4c_rm_comments_dataset \
  --epochs 20 \
  --block_size 512 \
  --train_batch_size 32 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456  2>&1 > train.log