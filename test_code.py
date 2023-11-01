import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append("./")
from finetune import SupervisedDataset,preprocess
from utils import ModelUtils
import json
from tqdm import tqdm
from transformers.trainer_pt_utils import LabelSmoother
from utils import remove_continuous_duplicate_sentences 

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# 使用合并后的模型进行推理
model_name_or_path = "Qwen_model/Qwen/Qwen-7B"      # Qwen模型权重路径
adapter_name_or_path = "/root/autodl-tmp/output_qwen_stage2_1030"    # sft后adapter权重路径

# 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
load_in_4bit = False
# 生成超参配置
max_new_tokens = 500
top_p = 0.9
temperature = 0.2
repetition_penalty = 1.0
device = 'cuda'
#test_paths = ["final_testset/QA_testset.jsonl","final_testset/RE_testset_free.jsonl","final_testset/NER_testset_free.jsonl"\
             # ,"final_testset/NER_testset_instruct.jsonl","final_testset/RE_testset_instruct.jsonl"]
test_paths = ["final_testset/MRD_testset.jsonl"]
# 加载模型
model = ModelUtils.load_model(
    model_name_or_path,
    load_in_4bit=load_in_4bit,
    adapter_name_or_path=adapter_name_or_path
).eval()
model.torch_dtype=torch.float32
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
def get_result_for_cmeee(response):
    if response == '':
        return []
    try:
        type_ents = response.strip('\n').split('\n')
        ner_results = list()
        for type_ent in type_ents:
            ner_type,type_results = type_ent.split('：')
            type_results = type_results.split("; ")
            for ent in type_results:
                ner_results.append(ner_type + ',' + ent)
    except:
        return []
    return ner_results

def get_result_for_cmeie(response):
    if response == '':
        return [],[]
    predic_triples = list()
    predic_r_triples = list()
    type_relations = response.split(";\n")
    try:
        for type_relation in type_relations:
            re_type, type_results = type_relation.split(":") 
            type_results = type_results.split(";")
            for type_result in type_results:
                type_result = type_result.strip('[').strip(']')
                [ent1,ent2] = type_result.split(',')
                predic_triple = "(" + ent1 + "," + ent2 + "," + re_type + ")"
                predic_r_triple = "(" + ent2 + "," + ent1 + "," + re_type + ")"
                predic_triples.append(predic_triple)
                predic_r_triples.append(predic_r_triple)
    except:
        return [],[]
    #print(predic_triples)
    return predic_triples,predic_r_triples
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
    #print(conversation)
    data_dict = preprocess([conversation], tokenizer, max_len=1024, test_flag = False,multiturn_flag=True,history_max_len=history_max_len)    
    #print(data_dict)
    input_ids = data_dict["input_ids"]
    #print(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.int).to(device=device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs, skip_special_tokens = True)
    response = remove_continuous_duplicate_sentences(response)
    history.append([message,response])
    return history,response
if __name__ == '__main__':
    for test_path in test_paths:
        test_name = test_path.split('.')[0].split("/")[1]
        test_output_path = "test_output/" + test_name + ".jsonl"
        if not os.path.exists("test_output"):
            os.mkdir("test_output")
        with open(test_path,'r') as test_file, open(test_output_path,'w') as test_output_file:
            lines = test_file.readlines()
            for line in tqdm(lines):
                results = dict()
                line = line.strip('\n')
                one_sample = json.loads(line)
                history = []
                conversation = one_sample["conversation"]
                conversation_id = one_sample["conversation_id"]
                if test_name == "MRD_testset":
                    results["answer"] = list()
                for conver in conversation:
                    human = conver["human"]
                    if test_name == "NER_testset_free" or test_name == "NER_ex_free":
                        new_human = "在下述文本中标记出医学实体：\n" + human
                        #print(new_human)
                    elif test_name == "RE_testset_free" or test_name == "RE_ex_free":
                        new_human = "实体关系抽取：\n" + human
                    else:
                        new_human = human
                    history,response = generate(message=new_human, history=history,max_new_tokens=max_new_tokens,temperature=temperature,top_p=top_p)
                    if test_name == "RE_ex_free" or test_name == "RE_testset_free" or test_name == "RE_testset_instruct":
                        response = response.replace(":","：")
                        response = response.replace(",",", ")
                        response = response.replace("];[","]; [")
                        try:
                            type_relations = response.split(";\n")
                            final_response = ''
                            for type_relation in type_relations:
                                final_response += type_relation + '\n'
                            response = final_response
                        except:
                            response = ""
                    conver["assistant"] = response
                    results["conversation_id"] = conversation_id
                    if test_name == "MRD_testset":
                        results["answer"].append(response)
                    else:
                        results["answer"] = response
                json.dump(results,test_output_file,ensure_ascii=False)
                test_output_file.write('\n')
    


                
        