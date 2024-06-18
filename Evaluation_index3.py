import os
# 禁用分词器并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from bert_score import score
from datasets import load_metric
import sacrebleu

# 加载模型和分词器
# model_name = "/data/WLX/gpt_huggingface_Poetry_finetuning1/HealthCareMagic-100k_inputoutput_epoch30"
# model = GPT2LMHeadModel.from_pretrained(model_name)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('/data/WLX/gpt_huggingface_Poetry_finetuning1/gpt-2')
model = GPT2LMHeadModel.from_pretrained('gpt-2')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 生成回答函数
def generate_answer(question, max_length=50):
    inputs = tokenizer.encode(question, return_tensors='pt').to(device)
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# 评价指标函数
def evaluate_bertscore(candidates, references):
    P, R, F1 = score(candidates, references, lang='en', verbose=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def evaluate_rouge(candidates, references):
    rouge = load_metric('rouge')
    results = rouge.compute(predictions=candidates, references=references)
    return results

def evaluate_bleu(candidates, references):
    bleu = sacrebleu.corpus_bleu(candidates, [references])
    return bleu.score

# 示例测试
question = "What is the capital of France?"
answer = generate_answer(question)
references = ["The capital of France is Paris."]
candidates = [answer]

# 计算评价指标
P, R, F1 = evaluate_bertscore(candidates, references)
rouge_results = evaluate_rouge(candidates, references)
bleu_score = evaluate_bleu(candidates, references)

# 输出结果
print(f"Question: {question}\nAnswer: {answer}")
print(f"BERTScore - Precision: {P}, Recall: {R}, F1: {F1}")
print(f"ROUGE Results: {rouge_results}")
print(f"BLEU Score: {bleu_score}")
