from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import crud
from app.database import SessionLocal, init_db
from app.routes import complaints, incidents, users
from app.services import classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with SessionLocal() as db:
        crud.seed_departments(db)
    yield


app = FastAPI(
    title="AirWatch API",
    description="Community pollution reporting, AI prioritisation and department accountability.",
    version="0.1.0", lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(users.router)
app.include_router(complaints.router)
app.include_router(incidents.router)


@app.get("/", tags=["meta"])
def health():
    return {"app": "AirWatch", "status": "ok", "ml_models_loaded": classifier.models_available()}
