#去除长字符段落的程序见F:\deep_learning\GPT_study\HealthCareMagic-100k_dataset_deal\中的py文件


# 将txt文件处理为需要的格式
input_texts = []
output_texts = []


# 定义两个空列表，用于存放Input和Output内容
input_texts = []
output_texts = []

# 读取文本文件内容
with open('HealthCareMagic-100k_1.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 遍历每一行内容
i = 0
while i < len(lines):
    # 判断是否为Input内容，以 "Input: " 开头的行认为是Input内容
    if lines[i].strip().startswith("Input:"):
        input_text = lines[i].strip()[len("Input:"):].strip()
        input_texts.append(input_text)
        # 向后查找Output内容，直到遇到空行为止
        j = i + 1
        output_text = ""
        z = j
        while z < len(lines) and lines[z].strip().startswith("Output:"):
            output_text += lines[z].strip()[len("Output:"):].strip()
            z += 1
        output_texts.append(output_text.strip())
        # 将索引 i 移至 Output内容之后的下一个行
        i = j
    else:
        i += 1
formatted_dataset = []
for input_text, output_text in zip(input_texts, output_texts):
    formatted_sample = f"{input_text}\t{output_text}\n"
    formatted_dataset.append(formatted_sample)

with open('HealthCareMagic-100k_3.txt', 'w', encoding='utf-8') as f_out:
    f_out.writelines(formatted_dataset)
# 打印提取结果
print("Input Texts:")
for text in input_texts:
    print(text)

print("\nOutput Texts:")
for text in output_texts:
    print(text)

print(len(input_texts))
print(len(output_texts))
print("对话段长度：" + str(len(output_texts)))