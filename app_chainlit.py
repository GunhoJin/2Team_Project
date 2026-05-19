import os
import chainlit as cl
import google.generativeai as genai
from dotenv import load_dotenv

# 1. 기존 환경 변수(.env) 파일에서 GOOGLE_API_KEY 로드
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini 클라이언트 설정
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    # 키가 없을 경우 터미널 및 UI에 경고를 띄우기 위한 안전 장치
    print(" WARNING: GOOGLE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 2. 사용자가 웹 페이지에 처음 접속했을 때 실행되는 함수
@cl.on_chat_start
async def start():
    if not GOOGLE_API_KEY:
        await cl.Message(content="`.env` 파일에 Google API 키가 설정되지 않아 서비스를 이용할 수 없습니다.").send()
        return

    # 세션 시작 시 Gemini 모델 인스턴스 생성 및 저장 (gemini-pro 또는 최신 버전에 맞게 수정 가능)
    try:
        model = genai.GenerativeModel("gemini-pro")
        cl.user_session.set("gemini_model", model)
        
        await cl.Message(
            content=" **2Team Chainlit Chatbot**\nGoogle API를 통해 답변을 생성합니다."
        ).send()
    except Exception as e:
        await cl.Message(content=f"Gemini 모델 로드 중 에러 발생: {e}").send()

# 3. 사용자가 채팅창에 메시지를 입력할 때마다 실행되는 함수
@cl.on_message
async def main(message: cl.Message):
    user_query = message.content
    model = cl.user_session.get("gemini_model")

    if not model:
        await cl.Message(content="AI 모델이 정상적으로 로드되지 않았습니다.").send()
        return

    # 화면에 타자 치듯 실시간 스트리밍으로 뿌려줄 빈 메시지 창 먼저 개설
    msg = cl.Message(content="")
    await msg.send()

    try:
        # [ RAG 확장 포인트] 
        # 나중에 노트북에서 가공한 원본 소스(ChromaDB context) 검색 코드가 이 자리에 들어옵니다.
        # 지금은 기존 API 작동 여부를 먼저 확인하기 위해 쿼리를 그대로 전달합니다.
        
        # Gemini API 스트리밍 요청 발송
        response = model.generate_content(user_query, stream=True)
        
        # Google API가 돌려주는 청크(글자 조각)를 받아 실시간으로 UI에 주입
        for chunk in response:
            if chunk.text:
                await msg.stream_token(chunk.text)
        
        # 스트리밍 출력이 끝나면 최종 메시지 업데이트 상태로 확정
        await msg.update()

    except Exception as e:
        msg.content = f"API 호출 중 에러가 발생했습니다:\n`{e}`"
        await msg.update()