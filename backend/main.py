from fastapi import FastAPI
from fastapi.responses import FileResponse

from backend.routers import router

app = FastAPI()

app.include_router(router)


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")