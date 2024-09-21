# Kaggle: LMSYS-Chatbot Arena Human Preference Predictions

## 背景

- **比赛简介**: 本次Kaggle竞赛旨在预测用户在两个由大语言模型(LLM)驱动的聊天机器人生成的回答中更偏好哪个。参赛者需要开发一个机器学习模型，基于提供的对话数据预测用户的偏好，以改进聊天机器人与人类的互动方式，使其更符合人类的偏好。

- **数据描述**: 比赛提供了一组来自Chatbot Arena的对话数据，这些对话由不同的LLM生成的回答组成。数据集包括55k条官方数据和33k条去重数据（总共88k），用于训练和测试模型。额外地，我们还从`ultrafeedback`数据集中采样了3万条数据，生成伪标签以增强训练数据集的多样性和规模。

- **目标**: 通过开发一个高精度的预测模型，帮助提升聊天机器人的用户体验，使其在实际应用中能够更好地符合用户的预期和偏好。

> 我们在本次竞赛中摘得铜牌（Top 6%）~[获奖证书](https://Cyccyyycyc.github.io/docs/AWARDkaggleLMSYS103.png)。

## 方法及代码讲解

我们的解决方案包括以下五个主要步骤：模型选择与测试、截断处理的Prompt设计、LoRA微调与优化、测试时数据增强(TTA)策略以及特殊情况的后处理。

### 1. 模型选择与测试

我们选择了**Gemma-2-9b-it**作为起始模型。通过对比测试发现，该模型比其他模型（如**Llama3 8b**、**Llama3.1 8b**）效果更好。这可能是因为比赛主办方的ChatBot-1M数据集在**Gemma-2**的后训练阶段被使用过，因此**Gemma-2**在这个任务中已经具备一定的适应能力。

- **模型架构**: 使用了`Gemma2ForSequenceClassification`模型，这是一个三分类任务的模型，用于预测哪个回答更符合用户偏好。
- **优化技术**: 采用了**LoRA**（低秩适配）技术，专门针对模型的某些部分进行微调。

LoRA微调的参数设置如下：

```python
freeze_layers: 0
lora_r: 64
lora_alpha: 16
lora_dropout: 0.05
lora_bias: "none"
lora_target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"
```

- **训练环境与时间**: 在4张A100 GPU上运行实验，每次实验的第一阶段约需10小时，第二阶段约需15小时。

### 2. 截断处理的Prompt设计

为了解决对话长度超过最大token限制的问题，我们设计了一个独特的prompt。该设计能够合理地截断最后一轮对话，确保prompt、response A和response B都有适当比例的展现，避免在截断时完全切除某个部分的情况。

- **具体实现**: 通过检测对话的token数量是否超过最大长度(`max_length`)，在必要时对每部分进行截断。
  
```python
def tokenize_cls_p3(example, tokenizer, max_length, is_train):
    input_ids = []
    attention_mask = []
    dot_tokens = tokenizer("......", add_special_tokens=False)["input_ids"]
    final_p_tokens = tokenizer("\n\n---\nWhich response is better? [A or B or tie]\nAnswer: ", add_special_tokens=False)["input_ids"]
    
    for ps, ras, rbs in zip(example['prompt'], example['response_a'], example['response_b']):
        one_input_ids = [tokenizer.bos_token_id]
        prev_tokens_num = 2 + len(final_p_tokens) # 2个token用于bos_token和eos_token
        
        for idx, (p, ra, rb) in enumerate(zip(ps, ras, rbs)):
            # 为每一轮对话生成token
            r_tokens  = tokenizer(f'\n\n## Round {idx+1}:' if idx else f'## Round {idx+1}:', add_special_tokens=False)["input_ids"]
            p_tokens  = tokenizer(f'\n### Prompt:\n{p}', add_special_tokens=False)["input_ids"]
            ra_tokens = tokenizer(f'\n\n### Response A:\n{ra}', add_special_tokens=False)["input_ids"]
            rb_tokens = tokenizer(f'\n\n### Response B:\n{rb}', add_special_tokens=False)["input_ids"]
            
            all_tokens_num = prev_tokens_num + len(r_tokens) + len(p_tokens) + len(ra_tokens) + len(rb_tokens)

            # 检查是否超过最大长度max_length
            if all_tokens_num > max_length:
                remain_tokens_num = max_length - prev_tokens_num - len(r_tokens) - 3*len(dot_tokens) 
                if remain_tokens_num >= 80:
                    # 对每个部分进行合理的截断
                    p_tokens  =  p_tokens[:int(remain_tokens_num*0.2)] + dot_tokens if len( p_tokens) > int(remain_tokens_num*0.2) else  p_tokens
                    ra_tokens = ra_tokens[:int(remain_tokens_num*0.4)] + dot_tokens if len(ra_tokens) > int(remain_tokens_num*0.4) else ra_tokens
                    rb_tokens = rb_tokens[:int(remain_tokens_num*0.4)] + dot_tokens if len(rb_tokens) > int(remain_tokens_num*0.4) else rb_tokens
                    one_input_ids += r_tokens + p_tokens + ra_tokens + rb_tokens
                break
            else:
                prev_tokens_num = all_tokens_num
                one_input_ids += r_tokens + p_tokens + ra_tokens + rb_tokens
        
        one_input_ids += final_p_tokens + [tokenizer.eos_token_id]
        one_attention_mask = [1] * len(one_input_ids)

        input_ids.append(one_input_ids)
        attention_mask.append(one_attention_mask)
    
    if is_train:
        labels = [0 if a_win else 1 if b_win else 2 for a_win, b_win, tie in zip(example['winner_model_a'], example['winner_model_b'], example['winner_tie'])]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    else:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
```

### 3. LoRA微调与优化

**LoRA**技术用于对模型进行微调。我们选择在bf16精度下进行微调，这样可以有效减少显存占用，从而允许使用更大的批次大小进行训练。通过这一方法，我们将模型的最后一层从传统的预测下一个token的任务改为分类任务。

- **目标模块**: 我们选择了`q_proj`、`k_proj`、`v_proj`、`o_proj`、`gate_proj`、`up_proj`和`down_proj`等模块作为LoRA的目标模块进行优化。  
- **LoRA参数设置**: 设置`lora_r`为64, `lora_alpha`为16, 并使用了`lora_dropout`为0.05，以确保微调效果和模型的稳定性。

### 4. 测试时数据增强（TTA）策略

在推理阶段，我们使用了测试时数据增强（TTA）策略。通过交换response A和response B进行数据增强，有效增加数据多样性，减少模型对单一数据顺序的依赖性，从而提高模型的稳健性和泛化能力。

- **具体方法**: 将response A和response B进行交换，得到两种不同的输入顺序，然后将两者的预测结果取平均值，作为最终的模型预测输出。

### 5. 特殊情况的后处理

为了进一步优化模型表现，我们在预测阶段对两种特殊情况进行了后处理：

1. **空回复处理**: 对于response A或response B为空的情况（如'[null]', '[]'等），我们将预测值固定为[0.04, 0.88, 0.08]。
2. **相同回复处理**: 对于response A和response B相同的情况，将预测值固定为[0.06, 0.06, 0.88]。

以下是后处理的具体代码：

```python
df2 = pd.read_csv('/kaggle/input/lmsys-chatbot-arena/test.csv')
df2['id'] = df2['id'].astype(str)

a_null_df = df2[(df2["response_a"]== '[null]') | (df2["response_a"]== '[]') | (df2["response_a"]== '[ ]') | (df2["response_a"]== '[  ]') | (df2["response_a"]== '[""]') | (df2["response_a"]== '["",""]')]
a_null_id_list = a_null_df["id"].tolist()
submission_df.loc[submission_df['id'].

isin(a_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.04, 0.88, 0.08]

b_null_df = df2[(df2["response_b"]== '[null]') | (df2["response_b"]== '[]') | (df2["response_b"]== '[ ]') | (df2["response_b"]== '[  ]') | (df2["response_b"]== '[""]') | (df2["response_b"]== '["",""]')]
b_null_id_list = b_null_df["id"].tolist()
submission_df.loc[submission_df['id'].isin(b_null_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.88, 0.04, 0.08]

same_a_b_df2 = df2[(df2["response_a"]==df2["response_b"])]
same_a_b_id_list = same_a_b_df2["id"].tolist()
submission_df.loc[submission_df['id'].isin(same_a_b_id_list), ['winner_model_a', 'winner_model_b', 'winner_tie']] = [0.06, 0.06, 0.88]
```

## 总结

通过上述五个关键步骤，我们成功开发并优化了一个高效的对话系统偏好预测模型，显著提升了模型在LMSYS-Chatbot Arena竞赛中的表现，最终摘得铜牌（Top 6%）。我们的模型方法证明了在处理类似大语言模型驱动的对话生成任务时，数据处理、模型微调、策略优化等方面的重要性和有效性。通过以上五个关键步骤，我们的模型方法展现了以下关键技术的有效性：

- **数据处理与扩展**：使用88k官方和去重数据，进行20折交叉验证，仅训练一折，并对`ultrafeedback`数据集进行伪标签扩展至10万+数据，增加数据集的多样性和规模。

- **Prompt设计与优化**：设计了独特的截断策略，在对话长度超出限制时合理分配显示prompt、response A、response B的比例，避免信息丢失。

- **LoRA微调技术**：采用LoRA技术对`Gemma2ForSequenceClassification`模型进行微调，专注于特定模块的优化，提高了模型在分类任务中的表现。

- **测试时数据增强 (TTA)**：在推理阶段通过交换response A和response B来增强数据多样性，减少模型对单一输入顺序的依赖性，提升模型的泛化能力。

- **特殊情况的后处理**：为处理空回复和相同回复情况，采用固定预测值的方法，优化模型在极端情况下的表现，提升log loss表现。

这些技术共同确保了模型在用户偏好预测任务中的高精度和稳健性，有效提升了模型在LMSYS-Chatbot Arena竞赛中的表现，最终取得了优异成绩。