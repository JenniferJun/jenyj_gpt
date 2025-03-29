import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")


function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


llm = ChatOpenAI(
    temperature=0.1,
).bind(
    function_call={
        "name": "create_quiz",
    },
    functions=[
        function,
    ],
)

def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
 
@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.

    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    The questions' difficulty should follow 'Difficulty' that has three types of easy, medium and hard.
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
    Your turn!
        
    Difficulty: {difficulty}
    Context: {context}
""",
        )
    ]
)
 
@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, difficulty):
    formatted_docs = format_docs(_docs)
    # 문서 데이터와 함께 체인을 구성합니다.
    chain = questions_prompt | llm
    # 체인을 호출하여 결과를 반환합니다. 
    return chain.invoke({"context": formatted_docs, "difficulty":difficulty})

@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    difficulty = st.sidebar.selectbox("Select the difficulty of the exam:", ["easy", "medium", "hard"])
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)
    


if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name, difficulty)
    response = response.additional_kwargs["function_call"]["arguments"]
    quiz_data = json.loads(response)
    with st.form("questions_form"): 
        answers = {}
        for i,question in enumerate(quiz_data["questions"]):
            value = st.radio(
                question["question"],
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
            answers[i] = value 
        submitted = st.form_submit_button('Submit', disabled='submitted' in st.session_state and st.session_state.submitted)
    if submitted: 
        score = 0
        total = len(quiz_data['questions'])
        # 답안 검증
        for i, question in enumerate(quiz_data['questions']):
            correct_answer = next(ans['answer'] for ans in question['answers'] if ans['correct'])
            if answers[i] == correct_answer:
                score += 1
        # 점수 출력
        st.subheader(f"You scored {score} out of {total}") 
        # 만점일 경우
        if score == total:
            st.session_state.submitted = True
            st.success("Congratulations! You scored a perfect score!")
            st.balloons()
        else:
        # 제출 버튼을 다시 표시
            st.session_state.submitted = False 