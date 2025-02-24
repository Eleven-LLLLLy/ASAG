# !/user/bin/env python3
# -*- coding: utf-8 -*-
import json
from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer
from LLM_reasoning import rank_criteria_with_llm,compare_answers_with_llm
from data_handle import *
import numpy as np
# File paths and configuration
json_file = 'data.json'
model_name = "/home/linshiyi/.cache/modelscope/hub/ZhipuAI/glm-4-9b-chat/"

# prompt

system_prompt_criteria_generating = '''
You are a professional educational evaluator who is responsible for evaluating student essays of 200-300 words on the importance of part-time work for college students.\
Now, you are asked to compare two student essays, decide which one is better, and explain your findings.\
Please base your evaluation on the following criteria and give 2-3 criteria to support your conclusion.\
Next I will give you an example, the final result should be returned in JSON format, strictly following the example format provided below.
<example>
given input:
answer 1:
"In my opinion, it is necessary for college students to have part-time job once their school grade or other condition become steady. Why? First, college students are old enough to be financially dependent. They are no more kids, and they have the responsibility to lessen the pressure from their parent. They have to understand how hard is working, thus, they can save and never waste their money. Second, they are no more students in a few years, so they have to judge what kind of job suit them through taking part-time jobs. Take me as an example. As a male, although I have to do the military responsibility first, and after one year then I have to work. I am very afraid now. Many of my classmates are worried about finding no job, and I may face the same situation after one year. One of my classmates declined a chance to be a teacher in the kindergarten, because she found herself having no interest in education. And some of my classmates felt sad and worried after the interview of a job, because they found themselves having no the skills or experiences that the boss wanted. So I think it's good to test college students' ability by having part-time job."
answer 2:
"In college, students want to find something to do to have the new experience and money. There are many ways to find something to do and the best one is part time job. It is good for students to do the part time job because it will make the students get more experiences, students will earn money, and they may have the opportunities to get the job in future. Firstly, it is good for students to do the part time jobs because students will get more experiences. When students have to do the jobs, they will face many situations from consumers and the boss of them. They will have more patients and learn more to live with others in societies. Secondly, students will earn more money while they are learning. Part time job makes money for students too. Student will know the value of money and how hard they find them. From this, they will know how to use money and they will have money if they have to use it. Finally, another reason why it is good for students to have a part time job is they might get more opportunities for their works in future. If students work, they will meet many people. In addition, if the agency finds that this student is good at working, they can remember your name and reserve this student for doing the job when this student graduates. All in all, the students might strongly believe that to do the part time job is necessary and important. Because part time job can give more experiences, make money, and provide more opportunities for student in the future."

You should output:
{
    "better_answer": "answer 1",
    "criteria":[
        "logic",
        "deep of content",
        "expression"
    ]
}
</example>

'''
system_prompt_rank = (
    "You are a helpful and detail-oriented assistant. You will be provided with a list of criteria extracted from a dataset. "
    "Your task is to analyze these criteria and rank them based on the following principles:\n\n"
    "1. Repetition: Criteria that appear more frequently across the dataset are considered more important.\n"
    "2. Importance: Criteria that are generally recognized as fundamental for evaluation should have higher priority.\n\n"
    "Provide a sorted list of the criteria, starting with the most important. "
    "Next I will give you an example, the final result should be returned in JSON format, strictly following the example format provided below."
    '''<example>
    given input:
     The criteria to rank are:"depth of content","clarity of argument","organization","logic","depth of content","expression"
    You should output:
    {
        "Rank_criteria": [
            "depth of content",
            "logic",
            "clarity of argument"
            "organization"
            "expression"
        ]
    }
    </example>
    '''
)

#Get Data
pair_data=load_formatted_answers()
criteria_list = load_criteria_from_json(json_file)
criteria_top_list=load_top_criteria()
real_scores=get_real_scores()
##评分标准矩阵获取
cri_matrix=criteria_matrix()

##特征矩阵以及特征向量获取
cri_eigenvectors_matrix=np.real(sigma(cri_matrix))
normal_cri_eigenvectors_matrix=normalize_vector(cri_eigenvectors_matrix)

matrix_3d = np.load('evaluation_under_criteria.npy')  # 假设文件名为 'matrix.npy'
# 获取三维矩阵的形状
m, n, _ = matrix_3d.shape  # m 是矩阵层数，n 是每个二维矩阵的维度

# 初始化结果矩阵，存储每个二维矩阵对应的最大特征向量
normal_scores_eigenvectors_matrix = np.zeros((m, n))  # 结果矩阵形状为 (m, n)

# 对每个二维矩阵计算最大特征值对应的特征向量并进行归一化
for i in range(m):
    normal_scores_eigenvectors_matrix[i, :] = normalize_vector(sigma(matrix_3d[i]))

# Main process
# rank_criteria_with_llm(criteria_list, system_prompt_rank, model_name, output_file="ranked_criteria.json")
compare_answers_with_llm(pair_data,criteria_top_list,model_name, output_file="evaluation_under_criteria.npy")
final_scores=np.dot(normal_cri_eigenvectors_matrix,normal_scores_eigenvectors_matrix.T)
print(final_scores)

##一致性指数计算
CI_result=calculate_consistency_index(final_scores, real_scores)
print(CI_result)
