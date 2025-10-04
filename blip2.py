"""
BLIP-2 Demo - 視覺問答與圖像描述生成
功能：
1. 自動生成圖片描述 (Image Captioning)
2. 回答關於圖片的問題 (Visual Question Answering)
"""

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

print(f"PyTorch 版本: {torch.__version__}")
print(f"MPS 是否可用: {torch.backends.mps.is_available()}")
print(f"MPS 是否已建置: {torch.backends.mps.is_built()}")

# 載入模型（使用較小的版本，適合快速測試）
print("正在載入 BLIP-2 模型...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# 如果有 GPU 就用 GPU
# 檢測可用設備
# if torch.backends.mps.is_available():
#     device = "mps"
#     print("✅ 使用 Apple Silicon GPU (MPS)")
# elif torch.cuda.is_available():
#     device = "cuda"
#     print("✅ 使用 NVIDIA GPU (CUDA)")
# else:
#     device = "cpu"
#     print("⚠️  使用 CPU")

device = "cpu"
model.to(device)

# 載入測試圖片
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

print("\n" + "="*60)
print("功能 1: 自動生成圖片描述")
print("="*60)

# 圖片描述生成（不需要輸入問題）
inputs = processor(images=image, return_tensors="pt").to(device, torch.float16 if torch.cuda.is_available() else torch.float32)

generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(f"生成的描述: {caption}\n")

print("="*60)
print("功能 2: 視覺問答 (VQA)")
print("="*60)

# 定義要問的問題
questions = [
    "Question: What animals are in the image? Answer:",
    "Question: How many cats are there? Answer:",
    "Question: What are the cats doing? Answer:",
    "Question: What color is the couch? Answer:",
    "Question: Where are the cats? Answer:"
]

for question in questions:
    # 處理問題和圖片
    inputs = processor(images=image, text=question, return_tensors="pt").to(
        device, torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # 生成答案
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    full_response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # 提取 "Answer:" 後面的內容
    if "Answer:" in full_response:
        answer = full_response.split("Answer:")[-1].strip()
    else:
        answer = full_response.replace(question, "").strip()
    
    print(f"Q: {question}")
    print(f"A: {answer}\n")
print("="*60)

"""
進階應用場景：
1. 智慧客服：自動理解用戶上傳的圖片並回答問題
2. 輔助視障人士：描述周圍環境
3. 電商應用：自動生成商品描述
4. 醫療影像：輔助醫生理解影像內容
5. 機器人導航：理解環境中的物體和場景

使用中文問答：
BLIP-2 主要訓練於英文，但可以：
1. 使用翻譯 API 先翻成英文
2. 或使用中文微調版本（如 mPLUG-Owl）

優化技巧：
1. 使用量化版本加速推理
2. 批次處理多張圖片
3. 快取模型避免重複載入
"""