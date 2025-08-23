from modelscope import snapshot_download
import os, shutil

# 目标目录
target_dir = r"../models/chatglm3-6b-32k"

# 下载模型到默认缓存
model_cache_dir = snapshot_download('ZhipuAI/chatglm3-6b-32k')

# 复制模型到目标目录
print(f"正在复制模型到 {target_dir} ...")
shutil.copytree(model_cache_dir, target_dir)

# 删除下载到缓存中的模型
print("正在删除缓存中的模型...")
shutil.rmtree(model_cache_dir)

print(f"✅ 模型已成功下载并移动到: {target_dir}")
