#!/bin/bash
source /opt/conda/bin/activate base
current_time=$(date +"%Y%m%d%H%M%S")
echo "ml-1m, ctr without embedding, current time: $current_time"
cd /mnt/data/0/LLM4Rec/
python train_ctr.py --model=dcn --pth_name=no_emb --is_long_tail=True --lr=0.001 --dataset=ml-1m --epoch=3 \
    --ctr_data_path=CTR/data/ml-1m/proc_data/ctr_data.csv \
    --meta_data_path=CTR/data/ml-1m/proc_data/ml-1m-meta.json

current_time=$(date +"%Y%m%d%H%M%S")
echo "ml-1m, dpo with true ground, current time: $current_time"
cd /mnt/data/0/LLM4Rec/LLaMA-Factory-main/
llamafactory-cli train examples/train_lora/ml-1m/llama3_lora_dpo_ml-1m_0.yaml
llamafactory-cli export examples/merge_lora/ml-1m/llama3_lora_dpo_ml-1m_0.yaml
rm -rf /mnt/data/0/LLM4Rec/LLaMA-Factory-main/saves/llama3-8b/lora/dpo/ml-1m/llama3_lora_llm4rec_dpo_ml-1m/

current_time=$(date +"%Y%m%d%H%M%S")
echo "ml-1m, embedding after 1 dpo, current time: $current_time"
cd /mnt/data/0/LLM4Rec/
python LLaMA-Factory-main/src/embedding/vllm_emb_oom.py --prompt_file="/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/ml_data_with_prompt.pkl" \
    --model="/mnt/data/0/LLM4Rec/llm_models/ml-1m/llama3_lora_llm4rec_dpo_ml-1m_0" \
    --emb_save_file="/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/all_ml-1m_emb_epoch_0.pkl"

current_time=$(date +"%Y%m%d%H%M%S")
echo "ml-1m, ctr with embedding after 1 dpo, current time: $current_time"
cd /mnt/data/0/LLM4Rec/
python train_ctr.py --model=dcn --is_vec=True --vec_dim=4096 --fusion_type=gate_logit_add --pth_name=with_emb_0 --is_long_tail=True --lr=0.001 --dataset=ml-1m --epoch=5 \
    --ctr_data_path=CTR/data/ml-1m/proc_data/ctr_data.csv \
    --meta_data_path=CTR/data/ml-1m/proc_data/ml-1m-meta.json \
    --emb_data_path=CTR/data/ml-1m/proc_data/all_ml-1m_emb_epoch_0.pkl

epoch=1
max_epochs=10
while [ $epoch -le $max_epochs ]
do
    echo "Epoch $epoch :"
    current_time=$(date +"%Y%m%d%H%M%S")
    echo "ml-1m, generating dpo data after $epoch dpo, current time: $current_time"
    cd /mnt/data/0/LLM4Rec/CTR/data/ml-1m/
    python ctr_data_for_dpo_processor.py --input_path=/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/llm_data_train.csv \
        --output_path=/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/dpo_ml-1m_$epoch.json \
        --model_path=/mnt/data/0/LLM4Rec/CTR/data/ml-1m/ctr_model_weights/dcn_with_emb_$((epoch-1))_best_model.pth \
        --emb_data_path=/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/all_ml-1m_emb_epoch_$((epoch-1)).pkl


    current_time=$(date +"%Y%m%d%H%M%S")
    echo "ml-1m, dpo with ctr predict, current time: $current_time"
    cd /mnt/data/0/LLM4Rec/LLaMA-Factory-main/
    llamafactory-cli train examples/train_lora/ml-1m/llama3_lora_dpo_ml-1m_$epoch.yaml
    llamafactory-cli export examples/merge_lora/ml-1m/llama3_lora_dpo_ml-1m_$epoch.yaml
    rm -rf /mnt/data/0/LLM4Rec/LLaMA-Factory-main/saves/llama3-8b/lora/dpo/ml-1m/llama3_lora_llm4rec_dpo_ml-1m/

    current_time=$(date +"%Y%m%d%H%M%S")
    echo "ml-1m, embedding after $((epoch+1)) dpo, current time: $current_time"
    cd /mnt/data/0/LLM4Rec/
    python LLaMA-Factory-main/src/embedding/vllm_emb_oom.py --prompt_file="/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/ml_data_with_prompt.pkl" \
        --model="/mnt/data/0/LLM4Rec/llm_models/ml-1m/llama3_lora_llm4rec_dpo_ml-1m_$epoch" \
        --emb_save_file="/mnt/data/0/LLM4Rec/CTR/data/ml-1m/proc_data/all_ml-1m_emb_epoch_$epoch.pkl"
    
    current_time=$(date +"%Y%m%d%H%M%S")
    echo "ml-1m, ctr with embedding after $((epoch+1)) dpo, current time: $current_time"
    cd /mnt/data/0/LLM4Rec/
    python train_ctr.py --model=dcn --is_vec=True --vec_dim=4096 --fusion_type=gate_logit_add --pth_name=with_emb_$epoch --is_long_tail=True --lr=0.002 --dataset=ml-1m --epoch=3 \
        --ctr_data_path=CTR/data/ml-1m/proc_data/ctr_data.csv \
        --meta_data_path=CTR/data/ml-1m/proc_data/ml-1m-meta.json \
        --emb_data_path=CTR/data/ml-1m/proc_data/all_ml-1m_emb_epoch_$epoch.pkl

    epoch=$((epoch + 1))
    sleep 5
done

echo "Done"


