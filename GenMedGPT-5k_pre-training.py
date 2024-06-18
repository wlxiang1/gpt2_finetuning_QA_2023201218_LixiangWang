import os
#在最开始进行设置用那一块GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
# from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter
# from datasets import load_dataset

#gpt给的新代码，无法运行，原因是找不到pandas.api.extensions库
'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

tokenizer.add_special_tokens({'cls_token': '[CLS]',
                              'mask_token': '[MASK]',
                              'pad_token': '[PAD]',
                              'sep_token': '[SEP]'
                              })
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('/data/WLX/gpt_huggingface_Poetry_finetuning1/gpt-2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

###
dataset = load_dataset('txt', data_files='/data/WLX/gpt_huggingface_Poetry_finetuning1/datadeal_and_debugtest/HealthCareMagic-100k_3.txt')

# 数据预处理函数
def preprocess_function(examples):
    inputs = [f"input: {i} output: {o}" for i, o in zip(examples['input'], examples['output'])]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs
#
# 应用预处理函数
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs_inputoutput',
)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

# 开始训练
trainer.train()

# 保存微调后的模型
model.save_pretrained("/data/WLX/gpt_huggingface_Poetry_finetuning1/finetunedmodel_add_inputandoutput_epoch30")
tokenizer.save_pretrained("/data/WLX/gpt_huggingface_Poetry_finetuning1/finetunedmodel_add_inputandoutput_epoch30")
'''


# 自定义数据集类
class QADataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.questions, self.answers = [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            # lines = f.readlines()
            # print(len(lines))

            # i = 0
            for line in f:
                # print("正在处理第"+str(i)+"行文本")
                # print("此行文本长度"+str(len(line)))
                result = line.strip().split('\t')
                self.questions.append(result[0])
                self.answers.append(result[1])
                # i = i+1
                # print(len(result[0]),len(result[1]))

        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        input_text = self.questions[idx] + " " + self.answers[idx]
        encodings = self.tokenizer(input_text, truncation=True, padding=True, return_tensors='pt')
        # print(encodings.input_ids.size(1))
        # target_shape = (1, 3100)
        target_shape = (1, 600)
        padding = (0, target_shape[1] - encodings.input_ids.size(1))
        # 进行填充
        encodings.input_ids = F.pad(encodings.input_ids, padding)
        # print(encodings.input_ids)
        return encodings.input_ids
#encodings中的三个值input_ids：这个键对应的值是经过 tokenization 后的输入文本所对应的 token ID 序列。每个 token ID 代表了 tokenizer 中的一个 token，它们是模型实际输入的表示
#token_type_ids：对于 GPT-2 这样的单输入模型，这个键对应的值通常是一个全为 0 的张量。在一些支持多输入的模型中，例如 BERT，token_type_ids 用于区分不同的输入序列。
#attention_mask：这个键对应的值是一个注意力掩码（attention mask）张量，用于指示模型在处理输入时哪些位置是有效的，哪些位置
##是填充的。通常，这个张量中的每个元素都是 1 或 0，1 表示对应位置是有效的，0 表示对应位置是填充的


# model_directory = "/data/WLX/gpt_huggingface_Poetry_finetuning1/HealthCareMagic-100k_inputoutput_epoch30"
model_directory = "/data/WLX/gpt_huggingface_Poetry_finetuning1/gpt-2"

# 加载GPT-2模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained(model_directory)

tokenizer.add_special_tokens({'cls_token': '[CLS]',
                              'mask_token': '[MASK]',
                              'pad_token': '[PAD]',
                              'sep_token': '[SEP]'
                              })
tokenizer.pad_token = tokenizer.eos_token



# model = GPT2LMHeadModel.from_pretrained('/data/WLX/gpt_huggingface_Poetry_finetuning1/gpt-2')
model = GPT2LMHeadModel.from_pretrained(model_directory)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# 加载数据集
dataset = QADataset('/data/WLX/gpt_huggingface_Poetry_finetuning1/datadeal_and_debugtest/HealthCareMagic-100k_3.txt', tokenizer)#dataset可以理解为一块儿内存
# max_length = max(len(sample) for sample in dataset)


data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

pre = time.time()
writer = SummaryWriter("/data/WLX/gpt_huggingface_Poetry_finetuning1/HealthCareMagic-100k_logs_epoch30")
# writer = SummaryWriter("HealthCareMagic-100k_logs_epoch30")
# 定义训练参数
total_train_step = 0
num_epochs = 30
learning_rate = 1e-4

# 定义优化器##
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 开始训练
for epoch in range(num_epochs):
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # outputs = model(input_ids=input_ids, labels=input_ids)
        loss, logits, _ = model(input_ids=input_ids, labels=input_ids)
        total_train_step =total_train_step + 1
        total_loss += loss
        # loss = outputs.loss
        loss.backward()
        optimizer.step()
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    original_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    if (epoch + 1) % 5 == 0:
                output_dir = f'fine_tuned_gpt2_modelHealthCareMagic-100k_epoch{int(epoch + 1)}'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                original_model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                print(f'Model saved to {output_dir}')


print('模型训练时间：', time.time() - pre)
writer.close()


