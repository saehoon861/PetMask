from fastapi import FastAPI
from fastapi.responses import FileResponse

#CSS, JavaScript, 이미지 같은 정적 파일을 브라우저가 요청할 수 있도록 제공하는 기능
from fastapi.staticfiles import StaticFiles 

from backend.routers import router


app = FastAPI()

app.include_router(router)

app.mount(
    "/frontend",
    StaticFiles(directory="frontend"),
    name="frontend"
)


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")
