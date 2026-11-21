"""Fills the DB with demo data (only if empty). Run:  python -m scripts.seed_demo
To start over, delete airwatch.db first."""
from app.database import SessionLocal, init_db
from app.services.demo import seed_demo

if __name__ == "__main__":
    init_db()
    with SessionLocal() as db:
        print("Seeded demo data." if seed_demo(db) else "Database already has data - nothing to do.")
