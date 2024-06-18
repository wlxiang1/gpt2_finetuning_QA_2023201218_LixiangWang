import json


# 读取数据集文件
with open('HealthCareMagic-100k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 构建微调模型输入格式的数据
formatted_data = []
for sample in data:
    input_text = sample["input"]
    output_text = sample["output"]
    formatted_sample = f"Input: {input_text}\nOutput: {output_text}"
    formatted_data.append(formatted_sample)

# 将格式化的数据保存到文件中
with open('HealthCareMagic-100k_1.txt', 'w', encoding='utf-8') as f:
    for sample in formatted_data:
        f.write(sample + '\n')
