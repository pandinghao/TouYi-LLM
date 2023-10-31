import gradio as gr
import re
# from TouyiModel import generate_response
from transformers import AutoTokenizer
import torch

import sys
sys.path.append("./")
from utils import ModelUtils
from finetune import make_supervised_data_module,preprocess

model_name_or_path = "Qwen_model/Qwen/Qwen-7B"      # Qwen模型权重路径
adapter_name_or_path = "/root/autodl-tmp/output_qwen_stage2_1030"     # sft后adapter权重路径
load_in_4bit = False
device = 'cuda:0'
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
if tokenizer.__class__.__name__ == 'QWenTokenizer':
    tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.bos_token_id = tokenizer.eod_id
    tokenizer.eos_token_id = tokenizer.eod_id
#
TOUYI_DES = """
<br>
<div align="center" style="font-size: 13pt;">
&nbsp;&nbsp;<img src="https://pic.imgdb.cn/item/651401a4c458853aef46f7f5.png" style="width: 13pt; display: inline-block;"> <a href="https://github.com/pandinghao/TouYi-LLM">Github</a>
<br>
"""


custom_css = """
#banner-image {
    margin-left: auto;
    margin-right: auto;
    width: 70px;
    height: 80px
}
"""

# 清空并提交文本框
def clear_and_save_textbox(message: str):
    return '', message

# 处理示例问题的函数
def process_example(message: str):
    generator = generate(message, [], 500, 0.10, 0.9)
    return '', generator

# retry按键回退对话记录
def delete_prev_fn(history):
    try:
        message, _ = history.pop()
    except IndexError:
        message = ''
    return history, message or ''

# 检查输入是否违规，即若不包含中文和英文则判为违规字符
def check_ch_en(text):
    ch_pat = re.compile(u'[\u4e00-\u9fa5]') 
    en_pat = re.compile(r'[a-zA-Z]')
    has_ch = ch_pat.search(text)
    has_en = en_pat.search(text)
    if has_ch and has_en:
        return True 
    elif has_ch:
        return True 
    elif has_en:
        return True 
    else:
        return False


# 删除历史消息中的系统提示语句，防止影响对话性能
def Delete_Specified_String(history):
    # 创建一个新的二维列表，用于存储不包含'你的输入无效'回答的问答对
    filtered_history = []
    # 遍历原始二维列表
    for pair in history:
        if pair[1] != '您的输入无效，请重新输入，谢谢！':
            filtered_history.append(pair)
    # 现在filtered_history中包含不包含'你的输入无效'回答的问答对
    return filtered_history


# 后处理，删除生成中的重复生成部分
def remove_continuous_duplicate_sentences(text):
    
    sentences = re.split(r'([。,；，\n])', text)
    
    # 初始化一个新的文本列表，用于存储去除连续重复句子后的结果
    new_sentences = [sentences[0]]  # 将第一个句子添加到列表中
    
    # 遍历句子列表，仅添加不与前一个句子相同的句子
    for i in range(2, len(sentences), 2):
        if sentences[i] != sentences[i - 2]:
            new_sentences.append(sentences[i - 1] + sentences[i])
    
    # 重新构建文本，使用原始标点符号连接句子
    new_text = ''.join(new_sentences)
    
    return new_text



# 生成函数，调用后端模型生成回复
def generate(
    message: str,
    history,
    max_new_tokens: int,
    temperature: float,
    top_p: float
):
    # 检查是否为空字符
    if not check_ch_en(message):
        generator = "您的输入无效，请重新输入，谢谢！"
        return  history+[(message.replace('\n', '\n\n'), generator)]
    # 生成超参配置n
    repetition_penalty = 1.0
    history_max_len = 1000  # 模型记忆的最大token长度
    conversation = list()
    for user_his, assist_his in history:
        conversation.append({"from": "user", "value": user_his})
        conversation.append({"from": "assistant", "value": assist_his})
    conversation.append({"from": "user", "value": message})
    #print("conversation:")
    #print(conversation)
    data_dict = preprocess([conversation], tokenizer, 1024, test_flag = False,multiturn_flag=True,history_max_len=history_max_len)    
    input_ids = data_dict["input_ids"]
    #print(input_ids)
    input_ids = torch.tensor(input_ids, dtype=torch.int).to(device=device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eos_token_id
        )
    #print("outputs")
    #print(outputs)
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs,skip_special_tokens = True)
    generator = remove_continuous_duplicate_sentences(response)
    #print(response)
    history.append((message,response))
    #print(history)
    # 检查history中是否包含了后处理的信息
    history = Delete_Specified_String(history)
    # generator = generate_response(message, history, max_new_tokens, temperature, top_p)
    #history.append((message,generator))
    return history




with gr.Blocks(css = custom_css) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown("# Touyi Sparse Finetuned Demo")
            gr.Markdown(TOUYI_DES)
            gr.Image("chat/test.jpg", elem_id="banner-image", show_label=False, container=False)
        with gr.Column():
            gr.Markdown("""### Touyi Sparse Finetuned Demo""")

            with gr.Group():
                chatbot = gr.Chatbot(
                    label = 'Chatbot',
                    bubble_full_width=False    
                )

                textbox = gr.Textbox(
                    container=False,
                    show_label=False,
                    placeholder="头一无敌",
                    lines=6
                )


            with gr.Row():
                retry_button = gr.Button('🔄  Retry', variant='secondary')
                undo_button = gr.Button('↩️ Undo', variant='secondary')
                clear_button = gr.Button('🗑️  Clear', variant='secondary')
                submit_button = gr.Button('🚩 Submit',variant='primary')



            gr.Examples(
            examples=[
                '最近肚子总是隐隐作痛，感觉胀胀的，吃下去的东西都没法吸收,请问是什么回事？',
                'What is the best treatment for sleep problems?'
            ],
            inputs=textbox,
            outputs=textbox,
            fn=process_example,
            cache_examples=False,
            label='Question Answering'
            )



            max_new_tokens = gr.Slider(
                label='Max new tokens',
                minimum=1,
                maximum=800,
                step=1,
                value=500,
                interactive=True,
            )
            temperature = gr.Slider(
                label='Temperature',
                minimum=0,
                maximum=1,
                step=0.05,
                value=0.10,
                interactive=True,
            )
            top_p = gr.Slider(
                label='Top-p (nucleus sampling)',
                minimum=0,
                maximum=1.0,
                step=0.1,
                value=0.9,
                interactive=True,
            )
            history_max_len = gr.Slider(
                label='History Max Length',
                minimum=500,
                maximum=1000,
                step=1,
                value=1000,
                interactive=True,
            )
    
    
    saved_input = gr.State()

    textbox.submit(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            max_new_tokens,
            temperature,
            top_p,
        ],
        outputs=chatbot,
        api_name=False,
    )

    button_event_preprocess = submit_button.click(
        fn=clear_and_save_textbox,
        inputs=textbox,
        outputs=[textbox, saved_input],
        api_name=False,
        queue=False,
    ).success(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            max_new_tokens,
            temperature,
            top_p,
        ],
        outputs=chatbot,
        api_name=False,
    )

    retry_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=generate,
        inputs=[
            saved_input,
            chatbot,
            max_new_tokens,
            temperature,
            top_p,
        ],
        outputs=chatbot,
        api_name=False,
    )

    undo_button.click(
        fn=delete_prev_fn,
        inputs=chatbot,
        outputs=[chatbot, saved_input],
        api_name=False,
        queue=False,
    ).then(
        fn=lambda x: x,
        inputs=[saved_input],
        outputs=textbox,
        api_name=False,
        queue=False,
    )

    clear_button.click(
        fn=lambda: ([], ''),
        outputs=[chatbot, saved_input],
        queue=False,
        api_name=False,
    )



demo.queue().launch(server_name="0.0.0.0", server_port=6006)
