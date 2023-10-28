# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
from datetime import datetime
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel
from accelerate.utils import DistributedType
from utils import find_all_linear_names
from torch.utils.tensorboard import SummaryWriter   # 通过注释这个来控制是否使用tensorboard
from my_trainer import TouYiTrainer, TouYiEvalPrediction

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

'''
用于添加各个参数的默认配置
'''
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")


@dataclass
class DataArguments:
    data_path: str = field(
        default="data/processed/train_cmedqa2_debug.json", metadata={"help": "Path to the training data."}
    )
    # eval_data_path: str = field(
    #     default="data/processed/valid_cmedqa2_debug.json", metadata={"help": "Path to the evaluation data."}
    # )
    eval_data_name: List[str] = field(
        metadata={"help": "Path to the evaluation data."}, default_factory=list
    )
    eval_data_path: List[str] = field(
        metadata={"help": "Path to the evaluation data."}, default_factory=list
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=128,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    train_from_former_checkpoint: str = None
    peft_path: str = None
    max_generated_tokens: int = field(default=500, metadata={"help": "max length of the generated tokens"})
    generation_top_p: float = field(default=0.9, metadata={"help": "top_p for generation"})
    generation_temperature: float = field(default=0.2, metadata={"help": "temperature for generation"})


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = None,
    test_flag: bool = False
) -> Dict:
    roles = {"user": "用户", "assistant": "助手"}

    #im_start = tokenizer.im_start_id
    #im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets,attention_mask = [], [],[]
    for i, source in enumerate(sources):
        '''
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]
        '''
        input_id, target = [tokenizer.bos_token_id], [IGNORE_TOKEN_ID]
        
        '''
        if system_message :
            system =  _system + tokenizer(system_message).input_ids + [tokenizer.eos_token_id]
            input_id += system
            target +=  [IGNORE_TOKEN_ID] * (len(system)) 
        '''
        if test_flag:
            target = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            #role = sentence["from"]
            if role == '用户' or (role == '助手' and not test_flag):
                _input_id = tokenizer(role).input_ids + nl_tokens + \
                    tokenizer(sentence["value"]).input_ids + [tokenizer.eos_token_id]
            #print(sentence["value"])
            input_id += _input_id
            if test_flag :
                if role == '助手':
                    _target = tokenizer(sentence["value"]).input_ids + [tokenizer.eos_token_id]
                    target += _target
            else:
                if role == '用户':
                    _target = [IGNORE_TOKEN_ID] * (len(_input_id))
                elif role == '助手' :
                    _target = [IGNORE_TOKEN_ID] * (len(tokenizer(role).input_ids) + 1) + \
                        tokenizer(sentence["value"]).input_ids + [tokenizer.eos_token_id]
                else:
                    raise NotImplementedError
                target += _target
        att_mask = [1] * len(input_id)
        if not test_flag:
            assert len(input_id) == len(target)
        if test_flag :
            input_id += tokenizer("助手").input_ids + nl_tokens
            att_mask = [1] * len(input_id)
            #target += [IGNORE_TOKEN_ID] * (len(tokenizer("助手").input_ids) + 1)
            #左padding
            #if len(input_id) < max_len:
                #input_id = [tokenizer.pad_token_id] * (max_len - len(input_id)) + input_id
                #target = [IGNORE_TOKEN_ID] * (max_len - len(target)) + target
        
        else:
            if len(input_id) < max_len:
                input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
                target += [IGNORE_TOKEN_ID] * (max_len - len(target))
                att_mask += [0] * (max_len - len(att_mask))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
        attention_mask.append(att_mask[:max_len])
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=attention_mask,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, test_flag = False):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len,test_flag = test_flag)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids = torch.tensor(self.input_ids[i], dtype=torch.int),
            labels = torch.tensor(self.labels[i], dtype=torch.int),
            attention_mask = torch.tensor(self.attention_mask[i], dtype=torch.int),
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, test_flag=False):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test_flag=test_flag

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.max_len, test_flag=self.test_flag)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_dataset = dict()
        for eval_data_name, eval_data_path in list(zip(data_args.eval_data_name, data_args.eval_data_path)):
            print(eval_data_name, eval_data_path)
            eval_json = json.load(open(eval_data_path, "r"))
            dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len,test_flag=True)
            eval_dataset[eval_data_name]=dataset
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
#todo:还需要改dataset
def compute_metrics(p: TouYiEvalPrediction):    
    ret = dict()    # 必须返回字典
    preds, labels, metric_key_prefix, tokenizer = p
    if metric_key_prefix.endswith("NER"):
        task = 'cmeee'
        precision, recall, f1 = eval_for_evalset(preds=preds,labels=labels,task=task,tokenizer=tokenizer)
        ret["P"] = precision
        ret["R"] = recall
        ret["F1"] = f1
    elif metric_key_prefix.endswith("RE"):
        task = 'cmeie'
        precision, recall, f1 = eval_for_evalset(preds=preds,labels=labels,task=task,tokenizer=tokenizer)
        ret["P"] = precision
        ret["R"] = recall
        ret["F1"] = f1
    else:
        pass    # 不需要计算指标的不进行处理，直接返回空字典

    return ret


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    training_args.logging_dir = "{}/{}".format(training_args.logging_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
    generation_config = transformers.GenerationConfig.from_pretrained(model_args.model_name_or_path)
    generation_config.max_new_tokens = training_args.max_generated_tokens
    generation_config.top_p = training_args.generation_top_p
    generation_config.temperature = training_args.generation_temperature
    training_args.generation_config = generation_config
    #print(training_args)
    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    #print(training_args.distributed_state.distributed_type)
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    training_args.ddp_find_unused_parameters = False
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        #quantization_config=GPTQConfig(
        #    bits=4, disable_exllama=True
        #)
        #if training_args.use_lora and lora_args.q_lora
        #else None,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        #llama不支持usefast
        trust_remote_code=True,
    )
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if training_args.use_lora:
        if lora_args.q_lora or 'chat' in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        #target_modules = find_all_linear_names(model)
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            #target_modules=target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        if training_args.peft_path :
            model = PeftModel.from_pretrained(model, training_args.peft_path, is_trainable=True)
        else:
            model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # Start trainner
    trainer = TouYiTrainer(
       model=model, tokenizer=tokenizer, args=training_args, compute_metrics=compute_metrics, **data_module
    )
    # trainer = Trainer(
    #     model=model, tokenizer=tokenizer, args=training_args, **data_module
    # )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)
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
def eval_for_evalset(preds,labels,tokenizer,task = None):
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
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = correct_cnt/all_pre_cnt
        recall = correct_cnt/gold_cnt
        f1 = 2*precision*recall/(precision + recall)
    return precision, recall, f1

if __name__ == "__main__":
    train()
