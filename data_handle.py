# # # !/user/bin/env python3
# # # -*- coding: utf-8 -*-
# # import csv
# # import os
# # import re
# # from itertools import permutations
# #
# # import numpy as np
# # from docx import Document
# # import pandas as pd
# # def get_score():
# #     file_path="/Volumes/Lin11/Data/Evaluation_data/ICNALE_GRA2_1/ICNALE_GRA_2_1.xlsx"
# #     sheet_index=5
# #     data=pd.read_excel(file_path,sheet_name=sheet_index)
# #     if 'Sample Info.1' in data.columns and 'Unnamed: 32' in data.columns:
# #         filter_condition = data['Sample Info.1'].str.contains('Essay', na=False)
# #         # 筛选符合条件的行，并选择对应列
# #         filtered_data = data.loc[filter_condition, ['Sample Info.1', 'Unnamed: 32']]
# #         return filtered_data
# #     else:
# #         print("列 T 或 AG 不存在，请检查数据列名")
# # def extract_number(filename):
# #     """从文件名中提取数字部分用于排序"""
# #     match = re.search(r'(\d{3})', filename)
# #     return int(match.group(1)) if match else -1  # 如果没有匹配到，则返回-1
# #
# # def get_text():
# #     folder_path = "/Volumes/Lin11/Data/Evaluation_data/ICNALE_GRA2_1/Rating Samples/ICNALE_GRA_Original_Essays"  # 替换为你的文件夹路径
# #
# #     # 用于存储所有文件的提取结果
# #     all_data = []
# #     files = [f for f in os.listdir(folder_path) if f.endswith('.docx') and not f.startswith('.')]
# #     sorted_files = sorted(files, key=extract_number)
# #     # 遍历文件夹中的所有文件
# #     for filename in sorted_files:
# #         file_path = os.path.join(folder_path, filename)
# #         try:
# #             doc = Document(file_path)
# #             text=[]
# #             for para in doc.paragraphs:
# #                 text.append(para.text)
# #             all_data.append(text)
# #         except Exception as e:
# #             print(f"Failed to read {filename}: {e}")
# #     return all_data
# #
# #
# # def select_and_pair_scores_texts(scores_df, texts, num_samples=10):
# #     """从分数和文本数据中选择一定数量的数据进行成对组合"""
# #     scores_array = scores_df['Unnamed: 32'].values  # 转化为NumPy数组
# #
# #     # 假设scores_df中的索引与texts列表中的顺序一致
# #     combined_df = pd.DataFrame({
# #         'Score': scores_array,
# #         'Text': texts
# #     })
# #
# #     # 按照分数进行等间隔采样，确保分数分布均匀
# #     indices = np.round(np.linspace(0, len(combined_df) - 1, num_samples)).astype(int)
# #     selected_pairs = combined_df.iloc[indices]
# #     print(selected_pairs)
# #     return selected_pairs
# # def create_all_possible_pairs(dataframe):
# #     """创建所有可能的有序对"""
# #     items = dataframe.values.tolist()
# #     # 使用permutations生成所有可能的有序对
# #     pairs = list(permutations(items, 2))
# #     return pairs
# # def save_pairs_to_csv(pairs, output_file='paired_samples.csv'):
# #     """将有序对保存到CSV文件"""
# #     with open(output_file, mode='w', newline='', encoding='utf-8') as file:
# #         writer = csv.writer(file)
# #         writer.writerow(['Score1', 'Text1', 'Score2', 'Text2'])
# #         for pair in pairs:
# #             writer.writerow([pair[0][0], pair[0][1], pair[1][0], pair[1][1]])
# #     print(f"有序对已保存至 {output_file}")
# #
# # def read_pairs_from_csv(input_file='paired_samples.csv'):
# #     """从CSV文件读取有序对"""
# #     pairs = []
# #     try:
# #         df = pd.read_csv(input_file)
# #         for _, row in df.iterrows():
# #             pairs.append(((row['Score1'], row['Text1']), (row['Score2'], row['Text2'])))
# #         print(f"成功从 {input_file} 读取有序对")
# #         return pairs
# #     except Exception as e:
# #         print(f"读取CSV文件时出错: {e}")
# #         return []
# # if __name__ == "__main__":
# #     all_data = get_text()
# #     all_score = get_score()
# #
# #     if all_score is not None and all_data:
# #         paired_samples = select_and_pair_scores_texts(all_score, all_data, num_samples=10)
# #         all_possible_pairs = create_all_possible_pairs(paired_samples)
# #         save_pairs_to_csv(all_possible_pairs)
# #         loaded_pairs = read_pairs_from_csv()
# #         for idx, pair in enumerate(loaded_pairs[:5], 1):
# #             print(f"Pair {idx}:")
# #             print(f"First item (Score: {pair[0][0]}, Text: {pair[0][1][:100]}...)")
# #             print(f"Second item (Score: {pair[1][0]}, Text: {pair[1][1][:100]}...)")
# #             print("-" * 40)
# #     else:
# #         print("未能获取到有效的分数或文本数据")
# # !/user/bin/env python3
# # -*- coding: utf-8 -*-
# import os
# import re
# from docx import Document
# import pandas as pd
# import numpy as np
# from itertools import permutations
#
#
# def get_score():
#     file_path = "/Volumes/Lin11/Data/Evaluation_data/ICNALE_GRA2_1/ICNALE_GRA_2_1.xlsx"
#     sheet_index = 5
#     data = pd.read_excel(file_path, sheet_name=sheet_index)
#
#     if 'Sample Info.1' in data.columns and 'Unnamed: 32' in data.columns:
#         filter_condition = data['Sample Info.1'].str.contains('Essay', na=False)
#         # 筛选符合条件的行，并选择对应列
#         filtered_data = data.loc[filter_condition, ['Sample Info.1', 'Unnamed: 32']]
#         return filtered_data
#     else:
#         print("列 Sample Info.1 或 Unnamed: 32 不存在，请检查数据列名")
#         return None
#
#
# def extract_number(filename):
#     """从文件名中提取数字部分用于排序"""
#     match = re.search(r'(\d{3})', filename)
#     return int(match.group(1)) if match else -1  # 如果没有匹配到，则返回-1
#
#
# def get_text():
#     folder_path = "/Volumes/Lin11/Data/Evaluation_data/ICNALE_GRA2_1/Rating Samples/ICNALE_GRA_Original_Essays"  # 替换为你的文件夹路径
#
#     all_data = []
#     files = [f for f in os.listdir(folder_path) if f.endswith('.docx') and not f.startswith('.')]
#     sorted_files = sorted(files, key=extract_number)
#
#     for filename in sorted_files:
#         file_path = os.path.join(folder_path, filename)
#         try:
#             doc = Document(file_path)
#             text = '\n'.join([para.text for para in doc.paragraphs])  # 合并所有段落为一个字符串
#             all_data.append(text)
#         except Exception as e:
#             print(f"Failed to read {filename}: {e}")
#     return all_data
#
#
# def select_and_pair_scores_texts(scores_df, texts, num_samples=10):
#     """从分数和文本数据中选择一定数量的数据进行成对组合"""
#     scores_array = scores_df['Unnamed: 32'].values  # 转化为NumPy数组
#
#     # 假设scores_df中的索引与texts列表中的顺序一致
#     combined_df = pd.DataFrame({
#         'Score': scores_array,
#         'Text': texts
#     })
#
#     # 按照分数进行等间隔采样，确保分数分布均匀
#     indices = np.round(np.linspace(0, len(combined_df) - 1, num_samples)).astype(int)
#     selected_pairs = combined_df.iloc[indices]
#
#     return selected_pairs
#
#
# def create_all_possible_pairs(dataframe):
#     """创建所有可能的有序对"""
#     items = dataframe.values.tolist()
#     # 使用permutations生成所有可能的有序对
#     pairs = list(permutations(items, 2))
#     return pairs
#
#
# def format_and_save_answers(pairs, output_file='formatted_answers.txt'):
#     """根据给定的格式整理文本并保存到文件"""
#     with open(output_file, mode='w', encoding='utf-8') as file:
#         for idx, pair in enumerate(pairs, 1):
#             file.write(f"answer 1:\n")
#             file.write(f'"{pair[0][1]}"\n')  # 第一个文本
#             file.write(f"answer 2:\n")
#             file.write(f'"{pair[1][1]}"\n')  # 第二个文本
#             file.write("-" * 40 + "\n\n")
#     print(f"格式化的答案已保存至 {output_file}")
#
#
# if __name__ == "__main__":
#     all_data = get_text()
#     all_score = get_score()
#
#     if all_score is not None and all_data:
#         paired_samples = select_and_pair_scores_texts(all_score, all_data, num_samples=10)
#         all_possible_pairs = create_all_possible_pairs(paired_samples)
#
#         # 格式化并保存文本答案
#         format_and_save_answers(all_possible_pairs)
#
#         # 示例：读取刚刚保存的答案文件
#         with open('formatted_answers.txt', 'r', encoding='utf-8') as file:
#             print(file.read())
#     else:
#         print("未能获取到有效的分数或文本数据")
import ast
import json
import re

import numpy as np
import pandas as pd

def load_formatted_answers(input_file='formatted_answers.txt'):
    """从文件中读取格式化的答案并返回为列表"""
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
            # 分割每个答案对
            answer_pairs = content.strip().split('-' * 40 + '\n\n')
        return answer_pairs
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []
def matrix_trans(pair_data):


    # 转化为矩阵形式
    rows = 10  # 答案对的数量
    cols = len(pair_data) // rows  # 每行的列数
    matrix = np.array(pair_data).reshape(rows, cols)

    print("矩阵形式：")
    print(matrix)
def parse_json(model_output):
    if type(model_output) is dict:
        return model_output
    elif type(model_output) is not str:
        model_output = str(model_output)
    try:
        model_output = model_output.replace("\n", " ")
        model_output = re.search('({.+})', model_output).group(0)
        model_output = re.sub(r"(\w)'(\w|\s)", r"\1\\'\2", model_output)
        result = ast.literal_eval(model_output)
    except (SyntaxError, NameError, AttributeError):
        return "ERR_SYNTAX"
    return result
def generate_comparison_prompt(criteria, answer):
    return (
        f"You are a helpful assistant. Compare two answers based on the following evaluation criterion: '{criteria}'.\n\n"
        f"{answer}\n\n"
        "Which answer better satisfies the criterion? Respond with one of the following options:\n"
        "'5' if Answer 1 is much better than Answer 2,\n"
        "'3' if Answer 1 is slightly better than Answer 2,\n"
        "'1' if both are equal,\n"
        "'1/3' if Answer 2 is slightly better than Answer 1,\n"
        "'1/5' if Answer 2 is much better than Answer 1."
    )
def load_criteria_from_json(input_file='data.json'):
    # Load the JSON file and extract all criteria
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten the list of criteria
    all_criteria = [criterion for item in data for criterion in item['criteria']]
    return all_criteria
def load_top_criteria(input_file='ranked_criteria.json'):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten the list of criteria

    all_criteria=data['Rank_criteria'][:10]
    return all_criteria
def criteria_matrix():
    with open('ranked_criteria.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    rank_criteria = data["Rank_criteria"]

    # 取前10个标准
    top_10_criteria = rank_criteria[:10]

    # 初始化评分矩阵
    n = len(top_10_criteria)
    A = np.ones((n, n))  # 初始化为1，因为a=a时评分为1

    # 根据规则填写矩阵
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # 自己与自己比较，跳过

            if rank_criteria.index(top_10_criteria[i]) < rank_criteria.index(top_10_criteria[j]):
                A[i, j] = 3  # a > b
                A[j, i] = 0.33  # 对称的关系
            elif rank_criteria.index(top_10_criteria[i]) > rank_criteria.index(top_10_criteria[j]):
                A[i, j] = 0.33  # a < b
                A[j, i] = 3  # 对称的关系
            else:
                A[i, j] = 1  # a == b
                A[j, i] = 1  # 对称的关系
    return A
def sigma(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # 获取最大特征值对应的索引
    max_eigenvalue_index = np.argmax(eigenvalues)

    # 返回最大特征值对应的特征向量
    return eigenvectors[:, max_eigenvalue_index]
def normalize_vector(v):
    total = np.sum(v)  # 计算向量的总和
    normalized_v = v / total  # 每个元素除以总和
    return normalized_v
def get_real_scores():
    df = pd.read_csv('paired_samples.csv')

    # 提取第一行和第1到第9行的得分
    # 假设需要提取的得分在 `Score1` 和 `Score2` 列中
    # 提取第一行和第1到第9行的得分，组合成一个长度为10的列表
    scores = [df.loc[0, 'Score1']]  # 第一个得分（第一行的Score1）

    # 追加第1-9行的得分
    for i in range(0, 9):  # 从第1行到第9行
        scores.append(df.loc[i, 'Score2'])  # 追加Score1

    return scores
def calculate_consistency_index(final_scores, real_score):
    n = len(final_scores)
    consistent_pairs = 0
    total_comparable_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            if (final_scores[i] > final_scores[j] and real_score[i] > real_score[j]) or \
                    (final_scores[i] < final_scores[j] and real_score[i] < real_score[j]):
                consistent_pairs += 1

    ci = consistent_pairs / total_comparable_pairs if total_comparable_pairs != 0 else 0
    return ci