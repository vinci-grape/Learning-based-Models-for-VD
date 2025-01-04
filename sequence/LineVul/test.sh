export CUDA_VISIBLE_DEVICES=0

python linevul_main.py \
  --model_name=model.bin \
  --output_dir=./storage/saved_models \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_test \
  --train_data_file=storage/dataset/train.json \
  --valid_data_file=storage/dataset/valid.json \
  --test_data_file=storage/dataset/test.json \
  --block_size 512 \
  --eval_batch_size 512