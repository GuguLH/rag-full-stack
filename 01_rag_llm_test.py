# 1 使用 nvidia-smi 查看本机GPU设备
import os

# 设置程序可见的显卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # 只使用第0和第1个显卡

# 2 控制大语言模型的输出的随机性的参数
"""
大语言模型预测下一个token时会先输出所有token的概率值，有不同的方法来控制选择哪一个token作为输出，主要以下4个参数
温度（Temperature）: 起到平滑调整概率的作用，temperature=1时，原始概率保持不变，temperature<1时，原来概率大的会变得更大（概率集中效果），temperature>1时,概率值越平均
Top-K: 模型输出是在概率在top-k的范围里随机选择一个，K值越大，选择范围越广，生成的文本越多样；K值越小，选择范围越窄，生成的文本越趋向于高概率的词。 k=1就是直接选择最高概率的token输出
Top-p: 通过累积概率来限定范围，top-p=0.5表示随机采样的范围是概率在前50%的tokens， top-p选择的tokens数是动态的
max-tokens: max-tokens参数指定了模型在停止生成之前可以生成的最大token（或词）数量
"""

import torch
import torch.nn.functional as f
import numpy as np

inputs = np.array([[2.0, 1.0, 0.1]])

temperature = 1
temperature = 0.1  # 概率集中
# temperature = 10  # 概率平滑

logits = torch.tensor(inputs / temperature)
softmax_scores = f.softmax(logits, dim=1)
print(f"temperature: {temperature} {softmax_scores.cpu().numpy()}")

# 3 modelscope 调用本地开源模型
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_path = "./models/Qwen2.5-0.5B-Instruct/"

# 加载本地模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    # device_map="cpu",
    # torch_dtype=torch.float16,  # 或 torch.bfloat16
)

# 分词器
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 模型的配置
gen_config = GenerationConfig.from_pretrained(model_path)

# 进行分词
ret = tokenizer("你好, are you ok")
print(ret)  # {'input_ids': [108386, 11, 525, 498, 5394], 'attention_mask': [1, 1, 1, 1, 1]}

# 108386就是你好n
ret = tokenizer.decode(108386)
print(ret)

# 模型参数
print(gen_config)


# 模型推理
def run_prompt(prompt: str, temperature=0.1, top_k=20, top_p=0.8, max_new_tokens=2048) -> str:
    gen_config.temperature = temperature
    gen_config.top_k = top_k
    gen_config.top_p = top_p

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print(text)

    # 输入token转换
    # 注意安装支持GPU的PyTorch
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    # uv add --extra-index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
    model_input = tokenizer([text], return_tensors="pt").to("cuda")
    # model_input = tokenizer([text], return_tensors="pt")

    # 进行推理
    generated_ids = model.generate(
        model_input.input_ids,
        max_new_tokens=max_new_tokens,
        generation_config=gen_config,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_input.input_ids, generated_ids)
    ]

    # 解码输出
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# 我的是RTX 1660s,显存不够,跑不起来,就算是改成cpu也不行,所以换模型 Qwen2.5-0.5B-Instruct
prompt_str = "Hello!"
ret = run_prompt(prompt_str)
print(ret)

# 4 huggingface transformers 调用本地开源模型
from transformers import AutoModel, AutoTokenizer

# 电脑GPU 显存不够,我无法调用
model_path = "./models/chatglm3-6b-32k/"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

model = AutoModel.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True,
)

model = model.half().eval()


def chat_glm(prompt, history=None, model=model, tokenizer=tokenizer):
    """
    非流式处理
    """
    if history is None:
        history = []
    response, history = model.chat(
        tokenizer,
        prompt,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        max_length=1024,
    )
    return response


def chat_glm_stream(prompt, history=None, model=model, tokenizer=tokenizer):
    """
    流式处理
    """
    if history is None:
        history = []
    for data in model.stream_chat(
            tokenizer,
            prompt,
            temperature=0.1,
            top_p=0.8,
            top_k=20,
            max_length=1024,
    ):
        yield data[0]


user_prompt = "你好,请介绍一下你自己"
ret = chat_glm(user_prompt)
print(ret)

# 流式输出
for res in chat_glm_stream(user_prompt):
    print(res, flush=True)

# 5 使用langchain
from langchain.llms.base import LLM

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun


class ChatGLM(LLM):
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        response, history = self.model.chat(self.tokenizer, prompt, history=[], temperature=0.1, top_p=0.8)
        return response

    @property
    def _llm_type(self) -> str:
        return "chatglm"


chatglm = ChatGLM(model, tokenizer)
ret = chatglm("你好")
print(ret)
