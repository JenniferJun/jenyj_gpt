from dotenv import load_dotenv
import os

# 환경 변수 로드
load_dotenv()

# 특정 환경 변수 가져오기
api_key = os.getenv("OPENAI_API_KEY")
another_variable = os.getenv("LANGCHAIN_API_KEY")

# 환경 변수 출력
print("OPENAI_API_KEY:", api_key)
print("LANGCHAIN_API_KEY:", another_variable)