# Long-HAT
Aim to long text understanding

run script
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python  -m torch.distributed.launch \
        --nproc_per_node 8 \
        --master_port 01234 \
        run_dp.py\
        --model r-f \ 
        --seed 42 \
        --lr 3e-5 \
        --max_train_length 512\
        --epochs 3 \
        --log_interval 10000 \
        --save_interval 10000 \
        --bs 32 \
        --layers 10-2 \
        --segments 2 \
        --plus 2 \
        --ap \
        --warm_step 7000 \
        --finetune
```
