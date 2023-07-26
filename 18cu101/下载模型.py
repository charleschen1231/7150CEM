import os
from transformers import AutoModel, AutoTokenizer

def main():
    # 设置Hugging Face API写访问令牌（write access token）为环境变量
    os.environ["HUGGINGFACE_CO_TOKEN"] = "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"

    # 设置模型名称
    model_name = 'distilbert-base-chinese'

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # 使用模型和分词器进行后续操作
    # ...

if __name__ == "__main__":
    main()
