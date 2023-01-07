from transformers import BertModel, BertConfig
import automodule

configuration = BertConfig()
model = BertModel(configuration)

m = automodule.AutoModule(model, (128, 768))
print(m)
