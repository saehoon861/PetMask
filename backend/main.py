from fastapi import FastAPI
from fastapi.responses import FileResponse
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
