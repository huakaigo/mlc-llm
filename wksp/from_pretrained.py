from transformers import AutoModelForCausalLM  # type: ignore[import]
model_path="../../base_model/vicuna-llama2-7b-watermark"
hf_model = AutoModelForCausalLM.from_pretrained(model_path)
# Get a list of parameters in advance, then delete the model to save memory
# print(hf_model.named_parameters().keys)
for k,v in hf_model.named_parameters():
    print(f"{k}")

for k,v in hf_model.named_parameters():
    print(f"{k: v}")
# param_list = [param for _, param in hf_model.named_parameters()]
# del hf_model