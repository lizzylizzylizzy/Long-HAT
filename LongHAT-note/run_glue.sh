CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python  -m torch.distributed.launch \
        --nproc_per_node 8 \
        --master_port 01234 \
        run_glue.py \
        --model_name_or_path roberta-base \
        --dataset_name imdb  \
        --do_train \
        --do_predict \
        --max_seq_length 32 \
        --per_device_train_batch_size 32 \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --output_dir ../imdb/