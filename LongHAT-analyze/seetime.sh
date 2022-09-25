
# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model rob \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 


CUDA_VISIBLE_DEVICES=2 \
python  -m torch.distributed.launch \
        --nproc_per_node 1 \
        --master_port 06233 \
        run_dp_seetime.py\
        --model r-f \
        --seed 42 \
        --lr 3e-5 \
        --max_train_length 1024\
        --epochs 1 \
        --warm_step 400 \
        --segments 8 \
        --log_interval 5000 \
        --save_interval 10000 \
        --bs 1 \
        --plus 1 \
        --grad_step 1 \
        --layers 6-6 \
        --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 6-6 \
#         --segments 4 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 6-6 \
#         --segments 8 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 


# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 8-4 \
#         --segments 2 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 8-4 \
#         --segments 4 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 8-4 \
#         --segments 8 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 


# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 10-2 \
#         --segments 2 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 10-2 \
#         --segments 4 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 10-2 \
#         --segments 8 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 6-6 \
#         --segments 16 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 8-4 \
#         --segments 16 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 

# CUDA_VISIBLE_DEVICES=2 \
# python  -m torch.distributed.launch \
#         --nproc_per_node 1 \
#         --master_port 06233 \
#         run_dp_seetime.py\
#         --model r-f \
#         --seed 42 \
#         --lr 3e-5 \
#         --max_train_length 1024\
#         --epochs 1 \
#         --warm_step 400 \
#         --log_interval 5000 \
#         --save_interval 10000 \
#         --bs 4 \
# 	    --layers 10-2 \
#         --segments 16 \
#         --plus 1 \
# 	    --grad_step 1 \
#         --finetune 