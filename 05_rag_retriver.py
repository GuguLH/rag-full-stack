# langchain使用
# 1 prompt template
from langchain_core.prompts import PromptTemplate

from provider import ModelProvider

template = """
{our_text}
你能为上述内容创建一个包含 {wordsCount} 个词的推文吗？
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["our_text", "wordsCount"]
)

final_prompt = prompt.format(our_text="我喜欢旅行，我已经去过6个国家。我计划不久后再去几个国家。",
                             wordsCount='3')

print("=" * 100)
print("prompt:")
print(final_prompt)

print("=" * 100)
print("answer:")
llm = ModelProvider.get_instance()
# resp = llm.invoke(final_prompt)
# print(resp.content)

# 少样本示例
from langchain_core.prompts import FewShotPromptTemplate

examples = [{'query': '什么是手机？',
             'answer': '手机是一种神奇的设备，可以装进口袋，就像一个迷你魔法游乐场。\
             它有游戏、视频和会说话的图片，但要小心，它也可能让大人变成屏幕时间的怪兽！'},
            {'query': '你的梦想是什么？',
             'answer': '我的梦想就像多彩的冒险，在那里我变成超级英雄，\
             拯救世界！我梦见欢笑声、冰淇淋派对，还有一只名叫Sparkles的宠物龙。'}]

example_template = """
Question: {query}
Response: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """你是一个5岁的小女孩，非常有趣、顽皮且可爱：
以下是一些例子：
"""

suffix = """
Question: {userInput}
Response: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["userInput"],
    example_separator="\n\n"
)

query = "房子是什么？"

real_prompt = few_shot_prompt_template.format(userInput=query)
print("=" * 100)
print("prompt:")
print(real_prompt)
print("=" * 100)
print("answer:")
# resp = llm.invoke(real_prompt)
# print(resp.content)

# 链式调用
chain = few_shot_prompt_template | llm

print("=" * 100)
print("answer:")
# resp = chain.invoke({"userInput": "房子是什么？"})
# print(resp.content)

from langchain_core.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
formate_instructions = output_parser.get_format_instructions()
prompt_template_cls = PromptTemplate(
    template="Provide 5 examples of {query}.\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": formate_instructions}
)
new_prompt = prompt_template_cls.format(query="房子是什么？")
print("=" * 100)
print(new_prompt)
