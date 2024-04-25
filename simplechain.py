
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

model_file = "Model/model-00001-of-00003.safetensors"


def load_llm(model_file):
    llm = CTransformers(
        model = model_file,
        model_type = "llama",
        max_new_token = 1024,
        temperature = 0.01
    )
    return llm

def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["question"])
    return prompt

#  Create simple chain

def create_simle_chain(prompt, llm):
    llm_chain = LLMChain(prompt, llm)
    return llm_chain

template = """
<|im_start|>system
Bạn là một trợ lí AI hữu ích. Hãy trả lời người dùng một cách chính xác.
<|im_end|>
<|im_start|>user
Hello world!<|im_end|>
<|im_start|>assistant"""

prompt = create_prompt(template)
llm = load_llm(model_file)
llm_chain = create_simle_chain(prompt, llm)


question = "who is the best football player"

respond = llm_chain.invoke({"question":question})
print(respond)