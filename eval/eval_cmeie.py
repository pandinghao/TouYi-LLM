import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer
import torch
import sys
sys.path.append("./")
from finetune import SupervisedDataset,preprocess
from utils import ModelUtils
import json

def get_cmeie_map (cmeie_des_path):
    with open(cmeie_des_path,"r") as input_file:
        des_dict = json.load(input_file)
        rel_to_id = dict()
        rel_type = des_dict["rel_type"]
        for re_type,_ in rel_type.items():
            rel_to_id[re_type] = len(rel_to_id)
    return rel_to_id
 
# 使用合并后的模型进行推理
model_name_or_path = "Qwen_model/Qwen/Qwen-7B"      # Qwen模型权重路径
adapter_name_or_path = "output_qwen/checkpoint-2200"     # sft后adapter权重路径

# 使用base model和adapter进行推理，无需手动合并权重
# model_name_or_path = 'baichuan-inc/Baichuan-7B'
# adapter_name_or_path = 'YeungNLP/firefly-baichuan-7b-qlora-sft'

# 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
load_in_4bit = True
# 生成超参配置
max_new_tokens = 500
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.0
eval_batch = 4
device = 'cuda:0'
# 加载模型
model = ModelUtils.load_model(
    model_name_or_path,
    load_in_4bit=load_in_4bit,
    adapter_name_or_path=adapter_name_or_path
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    padding_side='left',
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)

"相关（导致）:[失眠症,睡眠障碍];\n病因:[失眠症,沉醉状态];[失眠症,药物戒断]"
def eval(eval_path,despath,task = None):
    cmeie_map = get_cmeie_map(despath)
    with open(eval_path,"r"):
        eval_list = json.load(eval_path)
        all_sources = [example["conversations"] for example in eval_list]
        eval_sources = list()
        for i in range(len(all_sources)):
            eval_sources.append[all_sources[i]]
        preprocess    
def main():
    text = input('User：')
    while True:
        if tokenizer.__class__.__name__ == 'QWenTokenizer':
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token_id = tokenizer.eod_id
            tokenizer.eos_token_id = tokenizer.eod_id
        data_dict = preprocess([conversation], tokenizer, 1024, test_flag = True)
        print(data_dict)
        input_ids = data_dict["input_ids"].to(device)
        labels = data_dict["labels"]
        attention_mask = data_dict["attention_mask"]
        #input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        #bos_token_id = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
        #eos_token_id = torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long).to(device)
        #input_ids = torch.concat([bos_token_id, input_ids, eos_token_id], dim=1)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
        #print(model)
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        #print(len(outputs))
        response = tokenizer.decode(outputs,skip_special_tokens = False)
        #response = response.strip().replace(tokenizer.eos_token, "").strip()
        print("Touyi：{}".format(response))
        text = input('User：')


if __name__ == '__main__':
    main()