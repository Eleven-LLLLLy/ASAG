# !/user/bin/env python3
# -*- coding: utf-8 -*-
import json

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from data_handle import parse_json,generate_comparison_prompt
from tqdm import tqdm

def init_compare_answers_with_llm(system_prompt,pair_data):
    model_name = "/home/linshiyi/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat/"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data=[]
    for user_prompt in tqdm(pair_data):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            trust_remote_code=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        result = parse_json(response)
        print(result)
        data.append(result)

    with open('data.json', 'a+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
def rank_criteria_with_llm(criteria_list, system_prompt, model_name, output_file):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare the user prompt by combining all criteria
    user_prompt = f"The criteria to rank are:\n{', '.join(criteria_list)}"

    # Create messages for the chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        trust_remote_code=True
    )

    # Prepare input for the model
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode response
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Parse response and save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(response)

    print("Ranking completed. Output saved to", output_file)
def compare_answers_with_llm(pair_data,top_criteria, model_name, output_file):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    n = 10
    k = len(top_criteria)
    A_kij = np.zeros((k, n, n), dtype=int)  # 初始化 A_kij 矩阵
    for k_idx, criterion in enumerate(tqdm(top_criteria, desc="Processing criteria")):
        for i in range(n):
            temp=0
            for j in range(n):
                if i == j:
                    continue
                else:
                    system_prompt = generate_comparison_prompt(criterion,pair_data[temp])
                    temp+=1

                # 准备输入
                messages = [
                    {"role": "system", "content": system_prompt},
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    trust_remote_code=True
                )
                model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

                # 调用大模型生成输出
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=512,
                )
                generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # 解析输出
                try:
                    result = int(generated_text.strip())
                    A_kij[k_idx, i, j] = result
                    A_kij[k_idx, j, i] = -result  # 对称性
                except ValueError:
                    print(
                        f"Unexpected output for criterion '{criterion}' between answers {i} and {j}: {generated_text}")

    np.save(output_file, A_kij)
    print(f"Matrix saved to {output_file}")