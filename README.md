# 环境配置
pytroch 2.0以上版本
python 3.9以上版本
cuda 11.8 及以上

可以新建一个python虚拟环境，然后运行download.sh

# 数据处理
测试的时候有每个任务有两个分数，一种是用咱们自己的指令模板测，另一种是用评委写的指令模板测
## 数据原始格式
给出数据格式样例方便大家处理
### CMEDQA
    {
        "id":"0",
        "document_id":"0",
        "question_id":"24731702",
        "question":"不是说做b超对宝宝不好吗？那怀孕检查是不？不是说做b超对宝宝不好吗？那怀孕检查是不是越少越好。无麻烦解答，谢谢。",
        "type":"sqa",
        "choices":[

        ],
        "context":"",
        "answer":[
            "b超切实有一定的辐射，而且小孩比较的娇嫩，容易受辐射影响发育。宝宝尽量不要做b超，但是在胎儿期有母体的保护，所以不要担心，有必要的话一定要做。"
        ],
        "long_answer":[

        ]
    }
### CMEEE_V2 (实体识别NER任务)
    {
        "id":"0",
        "document_id":"0",
        "passages":[
            {
                "id":"0",
                "type":"sentence",
                "text":[
                    "（5）房室结消融和起搏器植入作为反复发作或难治性心房内折返性心动过速的替代疗法。"
                ],
                "offset":[
                    [
                        0,
                        40
                    ]
                ]
            }
        ],
        "entities":[
            {
                "id":"0",
                "type":"医疗程序",
                "text":[
                    "房室结消融"
                ],
                "offsets":[
                    [
                        3,
                        8
                    ]
                ],
                "normalized":[

                ]
            },
        ]
    }
### CMEIE_V2（关系抽取RE任务）
    {
        "id":"1",
        "document_id":"1",
        "passages":[
            {
                "id":"0",
                "type":"sentence",
                "text":[

                ],
                "offset":[
                    [
                        0,
                        173
                    ]
                ]
            }
        ],
        "entities":[
            {
                "id":"0",
                "type":"疾病",
                "text":[
                    "性早熟"
                ],
                "offsets":[
                    [
                        11,
                        14
                    ]
                ],
                "normalized":[
                ]
            }
        ],
        "events":[
        ],
        "relations":[
            {
                "id":"0",
                "type":"病理分型",
                "arg1_id":"0",
                "arg2_id":"7"
            }
        ]
    }
## 数据统一格式
- 指令泛化性
- 对话
- 实体识别，统一一个样本一个对话。
- 关系抽取也一样
- 一个样本对应一个指令（暂定）按比例划分，五个指令每个20%
    
        [
            {
                "id": "identity_0",
                "conversations": [
                {
                    "from": "user",
                    "value": "指令(给定标签范围) + '\qn' + 输入"
                },
                {
                    "from": "assistant",
                    "value": "你好,我是一个语言模型，我叫头一。"
                }
                ]
            }
        ]
# 训练

- qlora (hugginface trainer)
- 全参量 (deepspeed)(加分项4)
- 基于trainer重新搭建全部训练过程（针对加分项(1)）
# 后处理

- 针对各个任务的推理链路（用于测试集），包括错误格式筛除，回答解析指定格式（这部分训练集数据处理的时候也要考虑进去）

- 展示系统（汪志军负责，针对加分项（6））

# 任务评价指标


由各个任务数据处理负责人负责各自任务的评价指标,由我和谭整理带训练链路里（针对加分项（3））


# 学习资料
huggingface trainer https://huggingface.co/docs/transformers/main/en/main_classes/trainer
PEFT:https://zhuanlan.zhihu.com/p/649776098

# 常见trainer参数解析
--optim paged_adamw_32bit (优化器记录)通过设置--optim paged_adamw_32bit，来使用内存页面优化。这是transformers的Trainer中自带的一个超参数。大模型微调使用这个操作还是可以，但不是必须的。
--logging_dir ./tb_logs 
--logging_steps 50 
--evaluation_strategy steps 
--eval_steps 0.5/epoch (小于1是比例， 大于1是具体步长) 

