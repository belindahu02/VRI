import os
import torch
import h5py
from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/unixcoder-base"
# model_name = "microsoft/codebert-base"

output_folder = "CodeBERT/models"
output_name_pth = "unixcoder_model.pth"
output_name_h5 = "unixcoder_model.h5"
# output_name_pth = "codebert_model.pth"
# output_name_h5 = "codebert_model.h5"

os.makedirs(output_folder, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.save_pretrained(output_folder)

torch.save(model.state_dict(), os.path.join(output_folder, output_name_pth))
with h5py.File(os.path.join(output_folder, output_name_h5), 'w') as hf:
    for key, value in model.state_dict().items():
        hf.create_dataset(key, data=value.cpu().numpy())