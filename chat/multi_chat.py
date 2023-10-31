from transformers import AutoTokenizer
import torch

import sys
sys.path.append("./")
from utils import ModelUtils
from finetune import make_supervised_data_module,preprocess
#放到模型加载的部分
model_name_or_path = "/root/autodl-tmp/Qwen_model/Qwen/Qwen-7B"      # Qwen模型权重路径
adapter_name_or_path = "/root/autodl-tmp/output_qwen_stage2_1030/checkpoint-10386"     # sft后adapter权重路径
load_in_4bit = False
device = 'cuda:0'
model = ModelUtils.load_model(
    model_name_or_path,
    load_in_4bit=load_in_4bit,
    adapter_name_or_path=adapter_name_or_path
).eval()
max_new_tokens = 500
top_p = 0.9
temperature = 2

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    padding_side='left',
    # llama不支持fast
    use_fast=False if model.config.model_type == 'llama' else True
)
if tokenizer.__class__.__name__ == 'QWenTokenizer':
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.bos_token_id = tokenizer.eod_id
    tokenizer.eos_token_id = tokenizer.eod_id
#

def generate(
    message,
    history : list,
    max_new_tokens,
    top_p,
    temperature
):
    # 生成超参配置n
    repetition_penalty = 1.0
    history_max_len = 1000  # 模型记忆的最大token长度
    conversation = list()
    for user_his, assist_his in history:
        conversation.append({"from": "user", "value": user_his})
        conversation.append({"from": "assistant", "value": assist_his})
    conversation.append({"from": "user", "value": message})
    data_dict = preprocess([conversation], tokenizer, 1024, test_flag = False,multiturn_flag=True,history_max_len=history_max_len)    
    input_ids = data_dict["input_ids"]
    input_ids = torch.tensor(input_ids, dtype=torch.int).to(device=device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs,skip_special_tokens = False)
    history.append([message,response])
    return history


if __name__ == '__main__':
    history = []
    while True:
        message = input('User：')
        history = generate(message=message, history=history, max_new_tokens=max_new_tokens, top_p=top_p,temperature=temperature)
        print(history[-1][-1])