export CUDA_VISIBLE_DEVICES=8

allDataset=("vul4c_insert_comments_dataset" "vul4c_rm_comments_dataset" "vul4c_unexecuted_code_dataset" "vul4c_rename_identifier_dataset")

for dataset in "${allDataset[@]}";
do
  echo "begin dataset"
  echo $dataset;
  python linevul_main.py \
  --output_dir=./storage/saved_models/${dataset} \
  --model_type=roberta \
  --tokenizer_name=microsoft/codebert-base \
  --model_name_or_path=microsoft/codebert-base \
  --do_train \
  --do_test \
  --train_data_file=/data2/source/${dataset}/train_linevul.json \
  --valid_data_file=/data2/source/${dataset}/valid_linevul.json \
  --test_data_file=/data2/source/${dataset}/test_linevul.json \
  --epochs 20 \
  --block_size 512 \
  --train_batch_size 32 \
  --eval_batch_size 16 \
  --learning_rate 2e-5 \
  --max_grad_norm 1.0 \
  --evaluate_during_training \
  --seed 123456  2>&1 > train_${dataset}.log
done;

