from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import streamlit as st
import time 
 

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)
 
openapi_key = st.sidebar.text_input("OpenAI API KEY : ")

llm = ChatOpenAI(
    temperature=0.1,
    openai_api_key=openapi_key,
)

answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )
        

@st.cache_data(show_spinner="Loading website...")
def load_website(url):
    filter_exp = ["^https://developers\.cloudflare\.com/(ai-gateway/|vectorize/|workers-ai/).*"]
    
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url, 
        filter_urls=filter_exp,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter) 
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store.as_retriever()


st.title("SiteGPT (Assignment)")

if not openapi_key:
    st.error("Please enter your OpenAI API key to proceed.")

st.markdown(
    """
    Welcome! SiteGPT

    SiteGPT gathers Cloudflare's documentation 
    related to AI Gateway, Cloudflare Vectorize and Workers AI.

    You can ask questions about the gathered contents of a website.
            
    Start by writing the OpenAI API key on the sidebar and finishing the crawling URLs on the sitemap.
    
    Then Ask a question to the website.For example: 
    
    What is the price per 1M input tokens of the llama-2-7b-chat-fp16 model?
    What can I do with Cloudflare’s AI Gateway?
    How many indexes can a single account have in Vectorize?

    llama-2-7b-chat-fp16 모델의 1M 입력 토큰당 가격은 얼마인가요?
    Cloudflare의 AI 게이트웨이로 무엇을 할 수 있나요?
    벡터라이즈에서 단일 계정은 몇 개의 인덱스를 가질 수 있나요?

"""
)

if openapi_key:  
    retriever = load_website("https://developers.cloudflare.com/sitemap-0.xml")
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "\$")) 
 

