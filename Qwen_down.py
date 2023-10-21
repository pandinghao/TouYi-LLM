from modelscope.hub.snapshot_download import snapshot_download

model_name = "Qwen/Qwen-7B"
cache_dir = 'Qwen_model'
model_dir = snapshot_download(model_name, cache_dir=cache_dir)
