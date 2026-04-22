"""
setup_db.py — One-time database initialization.

Run this ONCE before starting the server:
    python setup_db.py

What it does:
  1. Enables the pgvector extension (for semantic LTM retrieval).
  2. Creates the ltm_facts table with proper indexes.
  3. Runs AsyncPostgresSaver.setup() to create LangGraph checkpoint tables.

Safe to re-run — all statements use IF NOT EXISTS / ON CONFLICT.
"""

import asyncio
import logging
import psycopg
from config import PG_CONN_STRING

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def setup_ltm_table():
    """Create ltm_facts table. Tries pgvector first, falls back to TEXT embedding."""
    log.info("Connecting to PostgreSQL...")
    with psycopg.connect(PG_CONN_STRING) as conn:
        with conn.cursor() as cur:

            # pgvector extension — optional but strongly recommended
            pgvector_available = False
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                conn.commit()
                pgvector_available = True
                log.info("✅ pgvector extension enabled.")
            except Exception as e:
                log.warning(f"⚠️  pgvector not available ({e}). Falling back to TEXT embeddings.")
                log.warning("   LTM retrieval will use keyword LIKE search instead of semantic search.")
                log.warning("   To enable pgvector: install the extension in your PostgreSQL instance.")
                conn.rollback()

            emb_col_type = "vector(768)" if pgvector_available else "TEXT"

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS ltm_facts (
                    id           SERIAL PRIMARY KEY,
                    user_id      TEXT NOT NULL DEFAULT 'default',
                    category     TEXT NOT NULL,
                    fact         TEXT NOT NULL,
                    embedding    {emb_col_type},
                    confidence   SMALLINT DEFAULT 1 CHECK (confidence BETWEEN 1 AND 3),
                    access_count INTEGER DEFAULT 0,
                    created_at   TIMESTAMPTZ DEFAULT NOW(),
                    updated_at   TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (user_id, fact)
                );

                CREATE INDEX IF NOT EXISTS idx_ltm_user
                    ON ltm_facts(user_id);

                CREATE INDEX IF NOT EXISTS idx_ltm_category
                    ON ltm_facts(user_id, category);

                CREATE INDEX IF NOT EXISTS idx_ltm_confidence
                    ON ltm_facts(user_id, confidence DESC);
            """)

            if pgvector_available:
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ltm_embedding
                        ON ltm_facts USING ivfflat (embedding vector_cosine_ops)
                        WITH (lists = 10);
                """)

            conn.commit()
            log.info("✅ ltm_facts table ready.")


async def setup_langgraph_tables():
    """Create LangGraph checkpoint tables using AsyncPostgresSaver.setup()."""
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        conn = await psycopg.AsyncConnection.connect(PG_CONN_STRING, autocommit=True)
        checkpointer = AsyncPostgresSaver(conn)
        await checkpointer.setup()
        await conn.close()
        log.info("✅ LangGraph checkpoint tables ready.")
    except ImportError:
        log.error("❌ langgraph-checkpoint-postgres not installed. Run: pip install langgraph-checkpoint-postgres")
        raise
    except Exception as e:
        log.error(f"❌ LangGraph table setup failed: {e}")
        raise


def main():
    log.info("=" * 50)
    log.info("  Personal Assistant — Database Setup")
    log.info("=" * 50)

    # 1. ltm_facts table
    try:
        setup_ltm_table()
    except Exception as e:
        log.error(f"❌ ltm_facts setup failed: {e}")
        log.error("   Is PostgreSQL running? Check: docker ps | grep pa-postgres")
        return

    # 2. LangGraph checkpoint tables
    try:
        asyncio.run(setup_langgraph_tables())
    except Exception as e:
        log.error(f"❌ LangGraph setup failed: {e}")
        return

    log.info("=" * 50)
    log.info("✅ Database setup complete. You can now run the server.")
    log.info("   Next step: python ingest.py   (if you have personal documents)")
    log.info("   Then:      uvicorn server:app --reload")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
