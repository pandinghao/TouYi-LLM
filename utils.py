from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
import bitsandbytes as bnb
import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class ModelUtils(object):

    @classmethod
    def load_model(cls, model_name_or_path, load_in_4bit=False, adapter_name_or_path=None):
        # 是否使用4bit量化进行推理
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        else:
            quantization_config = None

        # 加载base model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
            device_map='auto',
            quantization_config=quantization_config
        )

        # 加载adapter
        if adapter_name_or_path is not None:
            model = PeftModel.from_pretrained(model, adapter_name_or_path)

        return model
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_cmeie_map (cmeie_des_path):
    with open(cmeie_des_path,"r") as input_file:
        des_dict = json.load(input_file)
        rel_to_id = dict()
        rel_type = des_dict["rel_type"]
        for re_type,_ in rel_type.items():
            rel_to_id[re_type] = len(rel_to_id)
    return rel_to_id