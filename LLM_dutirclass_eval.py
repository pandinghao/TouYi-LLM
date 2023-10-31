import json
from collections import defaultdict


def str2dict(strlabel,task_name="NER"):
    split_list = strlabel.split('\n')
    golden_entities = {}
    for element in split_list:
        element = element.strip()
        if '：' in element:
            entity_type, entities_raw = element.split('：')
            entities = []
            if '; ' in entities_raw:
                entities_raw_list = entities_raw.split('; ')
                for entity in entities_raw_list:
                    if task_name =="RE":
                        entities.append(sort_entities(entity))
                    else:
                        entities.append(entity)
            else:
                if task_name == "RE":
                    entities.append(sort_entities(entities_raw))
                else:
                    entities.append(entities_raw)

            golden_entities[entity_type] = entities
        else:
            pass
    return golden_entities

def sort_entities(str_enti):
    try:
        o_ent = []
        h_ent, e_ent = str_enti.split(', ')
        o_ent.append(h_ent[1:])
        o_ent.append(e_ent[:-1])
        o_ent.sort()
        return str(o_ent)
    except:
        return str_enti


def com_prf(test_data_dir,pre_file_name,task_name='NER'):
    oupfile = open(pre_file_name, 'r', encoding='utf-8')
    oupfilelist = oupfile.readlines()
    all_gold_label = defaultdict(set)
    all_pred_label = defaultdict(set)
    all_gold = []
    all_pred = []

    with open(test_data_dir, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            line_dic = json.loads(line)
            label_dic = str2dict(line_dic["answer"],task_name=task_name)
            dic_str_pre = str2dict(json.loads(oupfilelist[idx])["answer"],task_name=task_name)

            # print('dic_str_pre',dic_str_pre)
            for key in label_dic.keys():
                if key in all_gold_label.keys():
                    pass
                else:
                    all_gold_label[key] = []
                for value in label_dic[key]:
                    all_gold_label[key].append(value+key+'idx'+str(idx))
                    all_gold.append(value+key+'idx'+str(idx))
            try:
                pred_dic_item = dic_str_pre
                for pre_key in pred_dic_item.keys():
                    if pre_key in all_pred_label.keys():
                        pass
                    else:
                        all_pred_label[pre_key] = []
                    for pre_value in pred_dic_item[pre_key]:
                        all_pred_label[pre_key].append(pre_value + pre_key+'idx'+str(idx))
                        all_pred.append(pre_value + pre_key+'idx'+str(idx))
            except:
                pass


    print('{0:^10}'.format(task_name+' 类别名称'),'\t','{0:^8}'.format('P'), '\t','{0:^8}'.format('R'), '\t','{0:^8}'.format('F1'))
    for type_name, true_entities in all_gold_label.items():
        pred_entities = all_pred_label[type_name]
        nb_correct = len(set(true_entities) & set(pred_entities))
        nb_pred = len(set(pred_entities))
        nb_true = len(set(true_entities))

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        print('{0:^10}'.format(type_name),'\t',format(p, '.4f'), '\t',format(r, '.4f'), '\t',format(f1, '.4f'))

    all_gold = []
    all_pred = []
    for i in all_gold_label:
        if all_pred_label[i] == set():
            all_pred_label[i] = []
        all_gold = all_gold+all_gold_label[i]
        all_pred = all_pred + all_pred_label[i]

    all_nb_correct = len(set(all_gold) & set(all_pred))
    all_nb_pred = len(set(all_pred))
    all_nb_true = len(set(all_gold))

    all_p = all_nb_correct / all_nb_pred if all_nb_pred > 0 else 0
    all_r = all_nb_correct / all_nb_true if all_nb_true > 0 else 0
    all_f1 = 2 * all_p * all_r / (all_p + all_r) if all_p + all_r > 0 else 0

    print('{0:^10}'.format('总类别'),'\t',format(all_p, '.4f'), '\t',format(all_r, '.4f'), '\t',format(all_f1, '.4f'))


if __name__ =="__main__":
    '''
    ner_test_data_dir: ner标准答案文件
    ner_pre_file_name：ner预测文件
    
    re_test_data_dir: re标准答案文件
    re_pre_file_name：re预测文件
    '''
    print("\n######## START NER TEST ########")
    ner_test_data_dir = "test_ouput/NER_ex_free.jsonl"
    ner_pre_file_name = 'results/NER_ex_pre.jsonl'
    com_prf(ner_test_data_dir, ner_pre_file_name,task_name='NER')
    print("######## END NER TEST ########\n")
    #
    print("\n######## START RE TEST ########")
    re_test_data_dir = "results/RE_ex_gold.jsonl"
    re_pre_file_name = 'final_testset/RE_ex_gold.jsonl'
    com_prf(re_test_data_dir, re_pre_file_name,task_name='RE')
    print("######## END RE TEST ########\n")