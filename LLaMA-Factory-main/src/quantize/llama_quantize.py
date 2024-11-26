from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.quantization import quantize_dynamic

model = AutoModelForCausalLM.from_pretrained("/home/llama3-8b-instruct").to('cpu')
model.eval()

model_quantized = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# save
model_path = "/home/quantized/llama3-8b-instruct-quantized.pth"
torch.save(model_quantized.state_dict(), model_path)

print("saved model ok!")

model_quantized = torch.load(model_path)
model_quantized.eval()


