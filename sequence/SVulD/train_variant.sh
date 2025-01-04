export CUDA_VISIBLE_DEVICES=5

allDataset=("vul4c_insert_comments_dataset" "vul4c_rm_comments_dataset" "vul4c_unexecuted_code_dataset" "vul4c_rename_identifier_dataset")

for dataset in "${allDataset[@]}";
do
  echo "begin dataset"
  echo $dataset;
  python run.py \
   --output_dir saved_models/${dataset} \
   --model_name_or_path microsoft/unixcoder-base-nine \
   --do_train \
   --train_data_file=/data2/source/${dataset}/train_svuld.json \
   --eval_data_file=/data2/source/${dataset}/valid_svuld.json \
   --num_train_epochs 40 \
   --block_size 400 \
   --train_batch_size 32 \
   --eval_batch_size 32 \
   --learning_rate 2e-5 \
   --max_grad_norm 1.0 \
   --seed 99 \
   --r_drop  2>&1 > train_${dataset}.log
done;

