CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python  -m torch.distributed.launch \
        --nproc_per_node 8 \
        --master_port 01234 \
        run_sc.py\
        --model rob \
        --seed 42 \
        --lr 1e-5 \
        --max_train_length 512\
        --epochs 5 \
        --log_interval 500 \
        --save_interval 10000 \
        --bs 8 \
        --plus 1\
        --grad_step 2 \
        --warm_step 200 \
        --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 8-4 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 8-4 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 8-4 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 8-4 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 8-4 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 8-4 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 8-4 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 8-4 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 2\
#         --segments 2\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --segments 2\
#         --warm_step 200 \
#         --finetune


# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 1.5e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 10-2 \
# #         --ap \
# #         --plus 1.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 2\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 1.5e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 10-2 \
# #         --ap \
# #         --plus 2.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --warm_step 200 \
#         --finetune


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 1\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 2e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 10-2 \
# #         --ap \
# #         --plus 1.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 2\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 2e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 10-2 \
# #         --ap \
# #         --plus 2.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 1\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 2.5e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 10-2 \
# #         --ap \
# #         --plus 1.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 2\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 2.5e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 10-2 \
# #         --ap \
# #         --plus 2.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 1\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 1e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 10-2 \
# #         --ap \
# #         --plus 1.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 2\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 1e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 10-2 \
# #         --ap \
# #         --plus 2.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 10-2 \
#         --ap \
#         --plus 3\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 1\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 1.5e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 6-6 \
# #         --ap \
# #         --plus 1.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 2\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 1.5e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 6-6 \
# #         --ap \
# #         --plus 2.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 3\
#         --warm_step 200 \
#         --finetune


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 1\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 2e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 6-6 \
# #         --ap \
# #         --plus 1.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 2\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 2e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 6-6 \
# #         --ap \
# #         --plus 2.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 3\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 1\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 2.5e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 6-6 \
# #         --ap \
# #         --plus 1.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 2\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 2.5e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 6-6 \
# #         --ap \
# #         --plus 2.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 2.5e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 3\
#         --warm_step 200 \
#         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 1\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 1e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 6-6 \
# #         --ap \
# #         --plus 1.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 2\
#         --warm_step 200 \
#         --finetune

# # CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# # python  -m torch.distributed.launch \
# #         --nproc_per_node 8 \
# #         --master_port 01234 \
# #         run_dp.py\
# #         --model r-f \
# #         --seed 42 \
# #         --lr 1e-5 \
# #         --max_train_length 256\
# #         --epochs 5 \
# #         --log_interval 500 \
# #         --save_interval 10000 \
# #         --bs 16 \
# #         --layers 6-6 \
# #         --ap \
# #         --plus 2.5\
# #         --warm_step 200 \
# #         --finetune

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 8 \
#         --master_port 01234 \
#         run_dp.py\
#         --model r-f \
#         --seed 42 \
#         --lr 1e-5 \
#         --max_train_length 256\
#         --epochs 5 \
#         --log_interval 500 \
#         --save_interval 10000 \
#         --bs 16 \
#         --layers 6-6 \
#         --ap \
#         --plus 3\
#         --warm_step 200 \
#         --finetune