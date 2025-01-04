export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python run.py \
    --output_dir saved_models/r_drop \
    --model_name_or_path microsoft/unixcoder-base-nine \
    --do_test \
    --test_data_file ./storage/dataset/test.json \
    --block_size 400 \
    --eval_batch_size 16 \
    --seed 99