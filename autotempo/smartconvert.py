from transformers import BertModel, BertConfig
import smartmodule
import torch

model = BertModel.from_pretrained("bert-base-uncased")

m = smartmodule.SmartModule(model, (512,), torch.int)
print(m)
