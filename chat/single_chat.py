import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer
import torch
import sys
sys.path.append("./")
from finetune import preprocess
from utils import ModelUtils
"""
单轮对话，不具有对话历史的记忆功能
"""


# 使用合并后的模型进行推理
model_name_or_path = "Qwen_model/Qwen/Qwen-7B"      # Qwen模型权重路径
adapter_name_or_path = "output_qwen/checkpoint-1125"     # sft后adapter权重路径

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


def main():
    text = input('User：')
    while True:
        conversation = [{"from": "user", "value": text}]
        print("conversation" + str(conversation))
        im_start = tokenizer.im_start_id
        tokenizer.pad_token_id = tokenizer.eod_id
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
        outputs = outputs.tolist()[0][len(input_ids[0]):]
        response = tokenizer.decode(outputs)
        #response = response.strip().replace(tokenizer.eos_token, "").strip()
        print("Touyi：{}".format(response))
        text = input('User：')


if __name__ == '__main__':
    main()