import json
import random
def merge_data(**kwargs):
    with open(kwargs["outputpath"],'w') as ouput_file:
        all_data = list()
        for task, path in kwargs.items():
            if task == "outputpath" or task == "shuffle":
                continue
            with open(path,'r') as input_file:
                data_list = json.load(input_file)
                for data in data_list:
                    data["id"] = str(data["id"])
                    if "conversation" in data.keys():
                        data["conversations"] = data["conversation"]
                        data.pop("conversation")
                all_data += data_list
        json.dump(all_data,ouput_file,indent=4,ensure_ascii=False)
    return all_data
final_dev = list()
cmeee = merge_data(outputpath = "data/processed/stage_2/cmeee.json", ner_eval_path = "data/processed/cmeee_v2-eval_set.json",ner_train_path = "data/processed/cmeee_v2-train_set.json")
#random.shuffle(cmeee)
cmeee_train = cmeee[200:]
cmeee_dev = cmeee[:200]
final_dev += cmeee_dev
cmeie = merge_data(outputpath = "data/processed/stage_2/cmeie.json", re_eval_path = "data/processed/cmeie_eval.json",re_train_path = "data/processed/cmeie_train.json")
#random.shuffle(cmeie)
cmeie_train = cmeie[200:]
cmeie_dev = cmeie[:200]
final_dev += cmeie_dev
cmedqa = merge_data(outputpath = "data/processed/stage_2/cmedqa.json", ner_eval_path = "data/processed/valid_cmedqa2.json",bioqa_train_path = "data/processed/train_cmedqa2.json")
random.shuffle(cmedqa)
cmedqa_train = cmedqa[200:]
cmedqa_dev = cmedqa[:200]
final_dev += cmedqa_dev
with open("data/processed/Touyi_data.json","r")as touyi_dia_file:
    touyi_list = json.load(touyi_dia_file)
    random.shuffle(touyi_list)
    touyi_d_train = touyi_list[200:]
    touyi_d_dev = touyi_list[:200]
    final_dev += touyi_d_dev
with open("data/processed/stage_2/stage2_train.json","w") as train_stage2_file, \
    open("data/processed/stage_2/stage2_dev.json","w") as dev_stage2_file, \
    open("data/processed/stage_2/cmeee_dev.json","w") as cmeee_dev_file, \
    open("data/processed/stage_2/cmeie_dev.json","w") as cmeie_dev_file, \
    open("data/processed/stage_2/cmedqa_dev.json","w") as cmedqa_dev_file, \
    open("data/processed/stage_2/touyi_dev.json","w") as touyi_dev_file:
    cmeee_cnt = 0
    cmeie_cnt = 0
    cmedqa_cnt = 0
    touyi_cnt = 0
    final_train = list()
    while(cmeee_cnt < len(cmeee_train) or cmeie_cnt < len(cmeie_train) or cmedqa_cnt < len(cmedqa_train) or touyi_cnt < len(touyi_d_train)):
        if cmeee_cnt + 15 < len(cmeee_train):
            final_train += cmeee_train[cmeee_cnt : cmeee_cnt + 15]
            cmeee_cnt += 15 
        else:
            final_train += cmeee_train[cmeee_cnt:]
            cmeee_cnt = len(cmeee_train)
        if touyi_cnt + 30 < len(touyi_d_train):
            final_train += touyi_d_train[touyi_cnt:touyi_cnt + 30]
            touyi_cnt+= 30
        if cmeie_cnt + 15 < len(cmeie_train):
            final_train += cmeie_train[cmeie_cnt:cmeie_cnt + 15]
            cmeie_cnt += 15
        else: 
            final_train += cmeie_train[cmeie_cnt:]
            cmeie_cnt = len(cmeie_train)
        if touyi_cnt + 23 < len(touyi_d_train):
            final_train += touyi_d_train[touyi_cnt:touyi_cnt + 23]
            touyi_cnt+= 23
        if cmedqa_cnt + 30 < len(cmedqa_train):
            final_train +=  cmedqa_train[cmedqa_cnt:cmedqa_cnt+ 30]
            cmedqa_cnt += 30
        else:
            final_train += cmedqa_train[cmedqa_cnt:]
            cmedqa_cnt = len(cmedqa_train)
        if touyi_cnt + 25 < len(touyi_d_train):
            final_train += touyi_d_train[touyi_cnt:touyi_cnt+25]
            touyi_cnt += 25
        else:
            final_train += touyi_d_train[touyi_cnt:]
            touyi_cnt = len(touyi_d_train)
        print(len(cmeie_train))
        print(len(cmeee_train))
        print(len(cmedqa_train))
        print(len(touyi_d_train))
        print ("111")
        print(cmeee_cnt)
        print(cmeie_cnt)
        print(cmedqa_cnt)
        print(touyi_cnt)
    print(len(final_train))
    json.dump(final_train, train_stage2_file,indent=4,ensure_ascii=False)
    json.dump(final_dev,dev_stage2_file,indent=4,ensure_ascii=False)
    json.dump(cmeee_dev,cmeee_dev_file,indent=4,ensure_ascii=False)
    json.dump(cmeie_dev,cmeie_dev_file,indent=4,ensure_ascii=False)
    json.dump(cmedqa_dev,cmedqa_dev_file,indent=4,ensure_ascii=False)
    json.dump(touyi_d_dev,touyi_dev_file,indent=4,ensure_ascii=False)
