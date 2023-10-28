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

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

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
adapter_name_or_path = "output_qwen_from8800/checkpoint-4400"     # sft后adapter权重路径

# 使用base model和adapter进行推理，无需手动合并权重
# model_name_or_path = 'baichuan-inc/Baichuan-7B'
# adapter_name_or_path = 'YeungNLP/firefly-baichuan-7b-qlora-sft'

# 是否使用4bit进行推理，能够节省很多显存，但效果可能会有一定的下降
load_in_4bit = False
# 生成超参配置
max_new_tokens = 500
top_p = 0.9
temperature = 0.35
repetition_penalty = 1.0
eval_batch = 1
device = 'cuda'
#eval_path = "data/processed/cmeie_eval_debug.json"
eval_path = "data/processed/cmeee_v2-eval_set.json"
task = "cmeee"
despath = "data/cmeee_v2-des.json"
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
    print(predic_triples)
    return predic_triples,predic_r_triples
def eval(eval_path, despath, task = None):
    #cmeie_map = get_cmeie_map(despath)
    with open(eval_path,"r") as evalfile:
        eval_list = json.load(evalfile)
        if tokenizer.__class__.__name__ == 'QWenTokenizer':
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token_id = tokenizer.eod_id
            tokenizer.eos_token_id = tokenizer.eod_id
        test_dataset = SupervisedDataset(raw_data=eval_list,tokenizer=tokenizer,max_len = 1024,test_flag = True)
        test_iters = DataLoader(dataset = test_dataset,
                    shuffle = False, 
                    drop_last = False,
                    batch_size = eval_batch)
    with torch.no_grad():
        all_pre_cnt = 0
        gold_cnt = 0
        correct_cnt = 0
        for tests in test_iters:
            input_ids = tests["input_ids"].to(device)
            labels = tests["labels"].to(device)
            outputs = model.generate(
                input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
                eos_token_id=tokenizer.eos_token_id
            )
            print("inputs")
            print(input_ids)
            print("outputs")
            outputs = outputs.tolist()
            print(outputs)
            new_outputs = []
            for i,output in enumerate(outputs):
                output = output[len(input_ids[i]):]
                new_outputs.append(output)
            outputs = new_outputs
            print(outputs)
            print("labels")
            print(labels)
            labels = labels.tolist()
            new_labels = []
            for i,label in enumerate(labels):
                new_labels.append(label)
            labels = new_labels
            print(labels)
            #print(len(outputs))
            responses = tokenizer.batch_decode(outputs,skip_special_tokens = True)
            label_responses = tokenizer.batch_decode(labels,skip_special_tokens=True)
            print(responses)
            print(label_responses)
            if task == "cmeee":
                for i,(response,label_response) in enumerate(zip(responses,label_responses)):
                    #print(response)
                    #print(label_response)
                    try:
                        ner_pred_results = get_result_for_cmeee(response = response)
                    except:
                        ner_pred_results = []
                    ner_labels = get_result_for_cmeee(response = label_response)
                    for ner_pred_result in ner_pred_results:
                        if ner_pred_result in ner_labels:
                            correct_cnt += 1
                    all_pre_cnt += len(ner_pred_results)
                    gold_cnt += len(ner_labels)
            if task == "cmeie":
                for i,(response,label_response) in enumerate(zip(responses,label_responses)):
                    #print(response)
                    #print(label_response)
                    try:
                        predic_triples,predic_r_triples = get_result_for_cmeie(response = response)
                    except:
                        predic_triples = []
                        predic_r_triples = []
                    label_triples,_ = get_result_for_cmeie(response = label_response)
                    for predic_triple in predic_triples:
                        if predic_triple in label_triples:
                            correct_cnt += 1 
                    for predic_r_triple in predic_r_triples:
                        if predic_r_triple in label_triples:
                            correct_cnt += 1
                    all_pre_cnt += len(predic_triples)
                    gold_cnt += len(label_triples)
            if all_pre_cnt == 0 or correct_cnt == 0 :
                f1 = 0
            else:
                precision = correct_cnt/all_pre_cnt
                recall = correct_cnt/gold_cnt
                f1 = 2*precision*recall/(precision + recall)
            print("f1")
            print(f1)
        print("final_f1")
        print(f1)
    return precision, recall, f1
def eval_for_evalset(preds,labels,task = None):
    gold_cnt = 0
    all_pre_cnt = 0
    correct_cnt = 0
    responses = tokenizer.batch_decode(preds,skip_special_tokens = True)
    label_responses = tokenizer.batch_decode(labels,skip_special_tokens=True)
    for i,(response,label_response) in enumerate(zip(responses,label_responses)):
        print(response)
        print(label_response)
        if task == "cmeee":
            for i,(response,label_response) in enumerate(zip(responses,label_responses)):
                #print(response)
                #print(label_response)
                try:
                    ner_pred_results = get_result_for_cmeee(response = response)
                except:
                    ner_pred_results = []
                ner_labels = get_result_for_cmeee(response = label_response)
                for ner_pred_result in ner_pred_results:
                    if ner_pred_result in ner_labels:
                        correct_cnt += 1
                all_pre_cnt += len(ner_pred_results)
                gold_cnt += len(ner_labels)
        if task == "cmeie":
            for i,(response,label_response) in enumerate(zip(responses,label_responses)):
                #print(response)
                #print(label_response)
                try:
                    predic_triples,predic_r_triples = get_result_for_cmeie(response = response)
                except:
                    predic_triples = []
                    predic_r_triples = []
                label_triples,_ = get_result_for_cmeie(response = label_response)
                for predic_triple in predic_triples:
                    if predic_triple in label_triples:
                        correct_cnt += 1 
                for predic_r_triple in predic_r_triples:
                    if predic_r_triple in label_triples:
                        correct_cnt += 1
                all_pre_cnt += len(predic_triples)
                gold_cnt += len(label_triples)
    if all_pre_cnt == 0 or correct_cnt == 0 :
        f1 = 0
    else:
        precision = correct_cnt/all_pre_cnt
        recall = correct_cnt/gold_cnt
        f1 = 2*precision*recall/(precision + recall)
    return precision, recall, f1

                    
                
            



                    



if __name__ == '__main__':
    precision, recall, f1 = eval(eval_path=eval_path,despath=despath,task=task)
    print(f1)