import re
import weaviate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores.weaviate import Weaviate
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from embedding import embedding_model
from llm import chat_model

# weaviate vector store
SELF_CONTENT = 'SelfContent'
SELF_CHAT_HISTORY = 'SelfChatHistroy'

# Create Database schema for content 
def create_context_schema_if_not_exist():
    class_name = SELF_CONTENT
    print('schema initialized with self chat history:', SELF_CONTENT)
    # Uncomment if you want to delete schema and create new
    # client.schema.delete_class(class_name)
    if client.schema.exists(class_name):
        print("schema already present")
        print('schema already exist.... continuing')
    else:
        schema_json = {'class': class_name,
                       'properties': [
                           {'dataType': ['text'], 'name': 'user_id'},
                           {'dataType': ['text'], 'name': 'message'},
                           {'dataType': ['text'], 'name': 'is_content', 'tokenization': 'word'},
                       ],
                       'vectorizer': 'none'
                       }
        client.schema.create_class(schema_json)
        print(f"schema {class_name} created successfully")
        
# Create DataBase schema for convo memory
def create_convo_schema_if_not_exist():
    class_name = SELF_CHAT_HISTORY
    print('schema initialized with self chat history:', SELF_CHAT_HISTORY)
    # Uncomment if you want to delete schema and create new
    # client.schema.delete_class(class_name)
    if client.schema.exists(class_name):
        print("schema already present")
        print('schema already exist.... continuing')
    else:
        schema_json = {'class': class_name,
                       'properties': [
                           {'dataType': ['text'], 'name': 'user_id'},
                           {'dataType': ['string[]'], 'name': 'convo_mem'},
                           {'dataType': ['text'], 'name': 'is_convo', 'tokenization': 'word'},
                       ],
                       'vectorizer': 'none'
                       }
        client.schema.create_class(schema_json)
        print(f"schema {class_name} created successfully")
        
WEAVIATE_URL = "http://af6409f3ccf7d403282a10f2f51f6eae-574440703.ap-south-1.elb.amazonaws.com"
client = weaviate.Client( url=WEAVIATE_URL )
create_context_schema_if_not_exist()
create_convo_schema_if_not_exist()

# saving the given .txt file to weaviate dB, (here we can include more file types as well)
def save_text_file(txt_file_path,user_id):
    loader = TextLoader(txt_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    for doc in splits:
        data_obj = {
            "user_id": user_id,
            "message": doc.page_content,
            "is_content": "true",
        }
        print("hi")
        text_embeddings=embedding_model.embed_documents([doc.page_content])[0]
        client.data_object.create(
            data_obj,
            SELF_CONTENT,
            consistency_level=weaviate.data.replication.ConsistencyLevel.ALL,
            vector=text_embeddings
        )

# Retriever for particular user
def retriever(user_id):
    vectorstore = Weaviate(client, index_name=SELF_CONTENT, text_key="message", embedding=embedding_model, by_text=False)
    text_filter = {
        'operator': 'And',
        'operands': [
            {
                'path': ["is_content"],
                'operator': 'Equal',
                'valueString': "true"
            },
            {
                'path': ["user_id"],
                'operator': 'Equal',
                'valueString': user_id
            }
        ]
    }
    retriever_ = vectorstore.as_retriever( search_type="mmr", search_kwargs={'k': 30, "where_filter": text_filter, 'lambda_mult': 0.25} )
    return retriever_

system_prompt = (
    "You are an expert in helping AI assistants manage their knowledge about a user and their operating environment."
    "Use the following pieces of retrieved context to answer the question."
    "Keep your answer more precise and concise, don't generate extra information this is very important."
    "\n\n"
    "{context}"
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.
Most Important point: If chat history is not relevant to formulate a standalone question, in such case return original question as it is"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat_model, qa_prompt,output_parser=StrOutputParser())

# rag chain unique to the user (as the retriever is unique to user)
def RAG_chain(user_id):
    history_aware_retriever = create_history_aware_retriever(
        chat_model, retriever(user_id=user_id), contextualize_q_prompt
    )
    rc = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rc

def reply(x,user_id):
    rag_chain = RAG_chain(user_id=user_id)
    # getting chat history before invoking the chain 
    chat_filter = {
        'operator': 'And',
        'operands': [
            {
                'path': ["is_convo"],
                'operator': 'Equal',
                'valueString': "true"
            },
            {
                'path': ["user_id"],
                'operator': 'Equal',
                'valueString': user_id
            }
        ]
    }
    ch = client.query.get(class_name=SELF_CHAT_HISTORY, properties=["convo_mem","_additional { id }"] ).with_where(chat_filter).do()
    list = ch['data']['Get'][SELF_CHAT_HISTORY]
    chat_history = []
    bol = False
    uuid = None
    if list != []: 
        bol = True
        chat_history = list[0]['convo_mem']
        uuid = list[0]['_additional']['id']
    ai = rag_chain.invoke({"input":x,"chat_history":chat_history})
    assi = re.findall(r'assistant\n(.*?)(?=system|user|$)', ai['answer'], re.DOTALL)
    ai_assi = assi[-1].strip()
    chat_history.extend(
        [
            "HumanMessage(content="+x+")",
            "AIMessage(content="+ai_assi+")"
        ]
    )
    # updating the chat history to the VDB
    if bol:
        client.data_object.update(
            data_object={"convo_mem":chat_history},
            class_name=SELF_CHAT_HISTORY,
            uuid=uuid,
        )
    else:
        data_obj = {
            "user_id": user_id,
            "convo_mem": chat_history,
            "is_convo": "true",
        }
        client.data_object.create(
            data_obj,
            SELF_CHAT_HISTORY,
            consistency_level=weaviate.data.replication.ConsistencyLevel.ALL,
        )
    return ai_assi

def get_chat_history(user_id):
    chat_filter = {
        'operator': 'And',
        'operands': [
            {
                'path': ["is_convo"],
                'operator': 'Equal',
                'valueString': "true"
            },
            {
                'path': ["user_id"],
                'operator': 'Equal',
                'valueString': user_id
            }
        ]
    }
    ch = client.query.get(class_name=SELF_CHAT_HISTORY, properties=["convo_mem","_additional { id }"] ).with_where(chat_filter).do()
    list = ch['data']['Get'][SELF_CHAT_HISTORY]
    chat_history = []
    if list != []: 
        chat_history = list[0]['convo_mem']
    return chat_history
