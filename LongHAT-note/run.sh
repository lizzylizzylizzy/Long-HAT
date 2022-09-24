CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python  -m torch.distributed.launch \
        --nproc_per_node 8 \
        --master_port 01223 \
        run_dp.py\
        --model rob \
        --seed 42 \
        --lr 1.5e-5 \
        --max_train_length 512\
        --epochs 3 \
        --log_interval 10000 \
        --save_interval 10000 \
        --bs 16 \
        --grad_step 2 \
        --warm_step 7000 \
        --finetune 