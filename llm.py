from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace

model_path = "/home/user/Desktop/semantic-search-langchain_agent/RAG/qwen2-0.5-Instruct"

llm = HuggingFacePipeline.from_model_id(
    model_id= model_path, 
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=512,
        do_sample=True,
        repetition_penalty=1.1
    ),
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True
)

pipeline = HuggingFacePipeline(
    model_kwargs={"quantization_config": quantization_config},
)

chat_model = ChatHuggingFace(llm=llm, model_id=model_path)
print("LLM chat model loaded...:).........................")