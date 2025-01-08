# !/user/bin/env python3
# -*- coding: utf-8 -*-
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from handle_file import load_formatted_answers, parse_json
from tqdm import tqdm
answer_pairs=load_formatted_answers()
model_name = "/root/chatglm3-6b-model"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

system_prompt = '''
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
data=[]
for answer_pair in tqdm(answer_pairs):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": answer_pair},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
    data.append(result)

with open('data.json', 'a+', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)