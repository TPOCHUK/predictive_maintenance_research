import os

# Unit tests replace DB calls and do not need a running PostgreSQL server.
os.environ["DATABASE_URL"] = "postgresql+psycopg://test:test@127.0.0.1:5432/test"
