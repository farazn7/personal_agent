#!/bin/bash

# fix_setup.sh — Fix pgvector and initialize the React UI
# Run from ~/project:  bash fix_setup.sh

set -e

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 1: Install pgvector in the Docker PostgreSQL"
echo "════════════════════════════════════════════════════════"

# Check if the container is running
if docker ps | grep -q "pa-postgres"; then
    echo "✅ pa-postgres container is running."
else
    echo "⚠️  pa-postgres not running. Trying to start it..."
    docker start pa-postgres 2>/dev/null || {
        echo ""
        echo "Container not found. Creating it now..."
        docker run --name pa-postgres \
            -e POSTGRES_USER=postgres \
            -e POSTGRES_PASSWORD=postgres \
            -e POSTGRES_DB=postgres \
            -p 5442:5432 \
            -d pgvector/pgvector:pg16
        echo "✅ Started pa-postgres with pgvector/pgvector:pg16 image."
        echo "   Waiting 3s for it to be ready..."
        sleep 3
    }
fi

# Check if the container uses the pgvector image already
IMAGE=$(docker inspect pa-postgres --format='{{.Config.Image}}' 2>/dev/null || echo "unknown")
echo "Container image: $IMAGE"

if echo "$IMAGE" | grep -q "pgvector"; then
    echo "✅ Container already uses pgvector image."
else
    echo ""
    echo "The container does NOT use the pgvector image."
    echo "We need to recreate it with pgvector/pgvector:pg16."
    echo ""
    echo "⚠️  This will DELETE the existing container (data is lost if not backed up)."
    echo "   Since this is a fresh setup, this is safe."
    echo ""
    read -p "Recreate with pgvector image? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo "Stopping and removing old container..."
        docker stop pa-postgres 2>/dev/null || true
        docker rm pa-postgres 2>/dev/null || true

        echo "Starting new container with pgvector support..."
        docker run --name pa-postgres \
            -e POSTGRES_USER=postgres \
            -e POSTGRES_PASSWORD=postgres \
            -e POSTGRES_DB=postgres \
            -p 5442:5432 \
            -d pgvector/pgvector:pg16

        echo "✅ New container started with pgvector."
        echo "   Waiting 4s for PostgreSQL to be ready..."
        sleep 4
    else
        echo "Skipping container recreation."
        echo "pgvector will not be available — LTM will use keyword search."
    fi
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 2: Re-run database setup"
echo "════════════════════════════════════════════════════════"
python setup_db.py

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 3: Check Node.js"
echo "════════════════════════════════════════════════════════"
if command -v node &>/dev/null; then
    echo "✅ Node.js $(node --version)"
else
    echo "❌ Node.js not found."
    echo "   Install it: https://nodejs.org  or  sudo apt install nodejs npm"
    exit 1
fi

if command -v npm &>/dev/null; then
    echo "✅ npm $(npm --version)"
else
    echo "❌ npm not found. Install Node.js from https://nodejs.org"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Step 4: Initialize React UI"
echo "════════════════════════════════════════════════════════"
cd ui/
echo "Installing dependencies..."
npm install
echo "✅ UI dependencies installed."

echo ""
echo "════════════════════════════════════════════════════════"
echo "  ✅ Setup complete!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "To start the backend:"
echo "    uvicorn server:app --host 0.0.0.0 --port 8000 --reload"
echo ""
echo "To start the frontend (separate terminal):"
echo "    cd ui && npm run dev"
echo ""
echo "Open: http://localhost:5173"
echo ""
