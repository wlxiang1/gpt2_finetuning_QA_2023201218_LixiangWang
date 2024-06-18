#测试的时候不需要训练，保存模型时没有使用tokenizer.save_pretrained(save_directory)命令，tokenizer需要使用gpt2的，
#tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import logging
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import keyboard
import random



# model_directory = "/data/WLX/gpt_huggingface_Poetry_finetuning1/fine_tuned_gpt2_modelHealthCareMagic-100k_2epoch"
# model_directory = "/data/WLX/gpt_huggingface_Poetry_finetuning1/fine_tuned_gpt2_modelHealthCareMagic-100k_epoch30"
model_directory = "/data/WLX/gpt_huggingface_Poetry_finetuning1/fine_tuned_gpt2_modelHealthCareMagic-100k_epoch30"
# model_directory = r"F:\\deep_learning\\gpt_huggingface_Poetry_finetuning\\fine_tuned_gpt2_modelHealthCareMagic-100k_epoch30"
tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model = GPT2LMHeadModel.from_pretrained(model_directory)
while True:
    input_text =input("请说明你的目前的身体有什么问题(使用英文)：")

    # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    special_tokens_dict = {'pad_token': '<PAD>'}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = '<PAD>'
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # 将输入文本编码成索引序列
    input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")

#Question:My family has a bad cold, and it is accompanied by a headache,why do i do this?
    # 使用模型进行文本生成
    # output = model.generate(input_ids, max_length=150, num_return_sequences=1, temperature=0.7)
    output = model.generate(
        input_ids,
        max_length=150,
        # num_beams=5,
        num_return_sequences=1,
        temperature=0.7,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    # 解码生成的文本序列
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

#
#
    # 计算PPL (Perplexity)
    def calculate_perplexity(text):
        tokens_tensor = tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            outputs = model(tokens_tensor, labels=tokens_tensor)
            # print(outputs)
            loss = outputs[0]
        return torch.exp(loss).item()

    perplexity = calculate_perplexity(generated_text)

    # 打印生成的文本
    print("医生的诊断结果:", generated_text)

    print("Perplexity:", perplexity)
    # if keyboard.is_pressed('esc'):
    #     print("按下了 ESC 键，程序结束")
    #     break

'''
temperature: 控制生成文本的随机性。较高的温度会生成更随机的文本。
repetition_penalty=2.0：会对已经生成的 token（即已经生成的词）施加一个惩罚，使得模型在选择下一个 token 时
，更倾向于选择不同的词，表示对于已经生成的 token，它们的 logits 会被除以 2，从而使模型在选择下一个 token 时更不倾向于选择这些已经生成过的 token。
top_k：top_k 的值通常设置为 1 到模型词汇表大小之间的一个整数。设置得过小可能会导致生成文本的多样性下降。较大的 top_k 值会增加生成文本
的多样性，但也可能增加生成不连贯或不合理文本的风险。
top_p：top_p 的值通常设置在 0.8 到 1.0 之间。影响：较低的 top_p 值（如 0.8 或 0.9）会限制候选 token 的范围，
从而提高生成文本的连贯性和相关性，但可能会减少生成文本的多样性。较高的 top_p 值（如 0.95 或 0.98）允许更多的候选 token，被采样的范围更广，从而增加生成文本的多样性。
'''