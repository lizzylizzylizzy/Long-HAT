# Long-HAT
## Research field
Aim to long text understanding

## Requirements
The dependencies can be installed by:
```
pip install -r requirement.txt
```

## File Description
-note (version with detailed annotation)  
-analyze (version with time consuming analysis for model components)

## Run Script
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python  -m torch.distributed.launch \
        --nproc_per_node 8 \
        --master_port <PORT> \
        run_dp.py\
        --model <MODEL_NAME> \ 
        --seed <SEED> \
        --lr <LR> \
        --max_train_length <MAX_SEQUENTCE_LENGTH>\
        --epochs <EPOCHS> \
        --log_interval <LOGGING_INTERVAL> \
        --bs <BATCH_SIZE> \
        --layers <LAYER_CONFIG> \
        --segments <SEGMENTS> \
        --plus <TIMES FOR LR IN TASK_SPECIFIC_LAYER> \
        --ap <ATTENTION_POOLER INSTEAD OF [CLS] TOKEN> \
        --warm_step <WARM_UPS> \
        --finetune
```
