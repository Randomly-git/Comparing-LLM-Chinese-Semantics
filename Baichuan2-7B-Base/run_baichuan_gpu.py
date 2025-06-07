import torch
from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
from contextlib import suppress

# 常量定义
MODEL_PATH = "/Baichuan2-7B-Base"  # 本地模型绝对路径
PROMPT = "明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？"

def load_model():
    """安全加载模型和tokenizer"""
    try:
        # 加载tokenizer（关闭fast模式兼容百川）
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,
            padding_side="left"  # 生成时左填充更稳定
        )

        # 自动设备映射（优先GPU）
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True  # 减少CPU内存占用
        ).eval()

        return tokenizer, model

    except Exception as e:
        print(f"加载失败，请检查: {e}")
        print(f"确保目录 {MODEL_PATH} 包含：config.json, pytorch_model.bin, tokenizer.model")
        exit(1)

def generate_response(tokenizer, model, prompt):
    # 输入处理（自动移动到模型所在设备）
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 流式输出配置
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=True,  # 不重复显示输入
        skip_special_tokens=True
    )

    # 科学生成参数
    generate_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 100,  # 严格控制长度
        "do_sample": True,
        "temperature": 0.6,     # 降低随机性
        "top_p": 0.7,           # 收紧候选词范围
        "top_k": 30,            # 限制候选词数量
        "repetition_penalty": 1.5,  # 重复抑制
        "num_return_sequences": 1,
        "eos_token_id": tokenizer.eos_token_id
    }

    # 抑制不必要的警告
    with suppress(RuntimeWarning, UserWarning):
        outputs = model.generate(**generate_kwargs)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    # 初始化
    tokenizer, model = load_model()

    # 生成优化提示词
    optimized_prompt = PROMPT

    # 执行生成
    print("生成结果：")
    response = generate_response(tokenizer, model, optimized_prompt)