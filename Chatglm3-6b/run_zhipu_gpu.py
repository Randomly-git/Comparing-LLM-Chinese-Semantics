from transformers import TextStreamer, AutoTokenizer, AutoModelForCausalLM
import torch

# 检查 CUDA 是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "/chatglm3-6b"  # 本地模型路径

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # 使用半精度节省显存
    device_map="auto",  # 自动分配到 GPU/CPU
).eval()

# 输入处理
prompt = ("明明明明明白白白喜欢他，可它就是不说，这句话中，到底谁喜欢谁？")
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 流式生成
streamer = TextStreamer(tokenizer)
outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=300)