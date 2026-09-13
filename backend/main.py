from fastapi import FastAPI
from fastapi.responses import FileResponse

#CSS, JavaScript, 이미지 같은 정적 파일을 브라우저가 요청할 수 있도록 제공하는 기능
from fastapi.staticfiles import StaticFiles 

from backend.routers import router


app = FastAPI()

# API 라우터를 먼저 등록합니다.
app.include_router(router)

# 나머지 모든 경로에 대해 frontend 디렉토리의 정적 파일을 제공합니다.
# html=True 설정을 통해 "/" 접속 시 자동으로 "index.html"을 찾습니다.
app.mount(
    "/",
    StaticFiles(directory="frontend", html=True),
    name="frontend"
)

# 기존의 @app.get("/")와 app.mount("/frontend")는 삭제하거나 아래 설정이 대체합니다.
