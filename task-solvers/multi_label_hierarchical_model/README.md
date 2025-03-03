## Run Multi label sequence classification with Huggingface Trainer

### Training and evaluation
#### Without descriptions

#### RoBERTa
```python
CUDA_VISIBLE_DEVICES=1 ./run.py --coarse_train ../../task_data/subtask2_hf/EN_coarse-grained/train.csv --fine_train ../../task_data/subtask2_hf/EN_fine-grained/train.csv --coarse_dev ../../task_data/subtask2_hf/EN_coarse-grained/dev.csv --fine_dev ../../task_data/subtask2_hf/EN_fine-grained/dev.csv --model_name_or_path FacebookAI/roberta-base --learning_rate 3e-5 --batch_size 16 --gradient_accumulation_steps 1 --epoch 100 --confidence_threshold 0.5 --train --predict
```
#### XLMRoBERTa
```python
CUDA_VISIBLE_DEVICES=1 ./run.py --coarse_train ../../task_data/subtask2_hf/EN_coarse-grained/train.csv ../../task_data/subtask2_hf/BG_coarse-grained/train.csv ../../task_data/subtask2_hf/HI_coarse-grained/train.csv ../../task_data/subtask2_hf/PT_coarse-grained/train.csv ../../task_data/subtask2_hf/RU_coarse-grained/train.csv --fine_train ../../task_data/subtask2_hf/EN_fine-grained/train.csv --coarse_dev ../../task_data/subtask2_hf/EN_coarse-grained/dev.csv --fine_dev ../../task_data/subtask2_hf/EN_fine-grained/dev.csv ../../task_data/subtask2_hf/EN_fine-grained/dev.csv ../../task_data/subtask2_hf/EN_fine-grained/dev.csv ../../task_data/subtask2_hf/EN_fine-grained/dev.csv ../../task_data/subtask2_hf/EN_fine-grained/dev.csv--model_name_or_path FacebookAI/roberta-base --learning_rate 3e-5 --batch_size 16 --gradient_accumulation_steps 1 --epoch 120 --confidence_threshold 0.5 --train --predict
```
