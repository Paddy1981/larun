#!/bin/bash
# =============================================================================
# LARUN MVP - Environment Setup Script
# =============================================================================
# This script sets up the development environment for LARUN
# Run this once when setting up a new development machine
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  LARUN MVP - Environment Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# Check prerequisites
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi
echo -e "${GREEN}  [OK] Docker installed${NC}"

# Check Docker Compose
if ! command -v docker compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi
echo -e "${GREEN}  [OK] Docker Compose installed${NC}"

# Check Node.js (optional for local web development)
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}  [OK] Node.js installed: ${NODE_VERSION}${NC}"
else
    echo -e "${YELLOW}  [WARN] Node.js not installed (optional for local web dev)${NC}"
fi

# Check Python (optional for local API development)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}  [OK] Python installed: ${PYTHON_VERSION}${NC}"
else
    echo -e "${YELLOW}  [WARN] Python not installed (optional for local API dev)${NC}"
fi

echo ""

# -----------------------------------------------------------------------------
# Create .env file if it doesn't exist
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Setting up environment file...${NC}"

ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/infrastructure/.env.example"

if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}  [SKIP] .env file already exists${NC}"
    echo "  If you want to recreate it, delete .env and run this script again"
else
    if [ -f "$ENV_EXAMPLE" ]; then
        cp "$ENV_EXAMPLE" "$ENV_FILE"

        # Generate secure secrets
        NEXTAUTH_SECRET=$(openssl rand -base64 32)
        JWT_SECRET=$(openssl rand -base64 32)

        # Replace placeholder secrets (macOS/Linux compatible)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|NEXTAUTH_SECRET=.*|NEXTAUTH_SECRET=$NEXTAUTH_SECRET|" "$ENV_FILE"
            sed -i '' "s|JWT_SECRET=.*|JWT_SECRET=$JWT_SECRET|" "$ENV_FILE"
        else
            # Linux
            sed -i "s|NEXTAUTH_SECRET=.*|NEXTAUTH_SECRET=$NEXTAUTH_SECRET|" "$ENV_FILE"
            sed -i "s|JWT_SECRET=.*|JWT_SECRET=$JWT_SECRET|" "$ENV_FILE"
        fi

        echo -e "${GREEN}  [OK] Created .env file with secure secrets${NC}"
        echo -e "${YELLOW}  [ACTION] Please update Stripe keys in .env${NC}"
    else
        echo -e "${RED}Error: .env.example not found at $ENV_EXAMPLE${NC}"
        exit 1
    fi
fi

echo ""

# -----------------------------------------------------------------------------
# Create necessary directories
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Creating directories...${NC}"

mkdir -p "$PROJECT_ROOT/models"
mkdir -p "$PROJECT_ROOT/output"
mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$PROJECT_ROOT/web"
mkdir -p "$PROJECT_ROOT/api"

echo -e "${GREEN}  [OK] Created necessary directories${NC}"
echo ""

# -----------------------------------------------------------------------------
# Pull Docker images
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Pulling Docker images...${NC}"

docker pull postgres:15-alpine
docker pull redis:7-alpine
docker pull node:20-alpine
docker pull python:3.11-slim

echo -e "${GREEN}  [OK] Docker images pulled${NC}"
echo ""

# -----------------------------------------------------------------------------
# Start services
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Starting database and redis services...${NC}"

cd "$PROJECT_ROOT/docker"
docker compose up -d db redis

echo -e "${GREEN}  [OK] Services started${NC}"
echo ""

# -----------------------------------------------------------------------------
# Wait for services to be healthy
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Waiting for services to be healthy...${NC}"

# Wait for PostgreSQL
for i in {1..30}; do
    if docker compose exec -T db pg_isready -U larun -d larun_db > /dev/null 2>&1; then
        echo -e "${GREEN}  [OK] PostgreSQL is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Error: PostgreSQL did not become ready in time${NC}"
        exit 1
    fi
    sleep 1
done

# Wait for Redis
for i in {1..30}; do
    if docker compose exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo -e "${GREEN}  [OK] Redis is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Error: Redis did not become ready in time${NC}"
        exit 1
    fi
    sleep 1
done

echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Services running:"
echo "  - PostgreSQL: localhost:5432"
echo "  - Redis: localhost:6379"
echo ""
echo "Next steps:"
echo "  1. Update Stripe keys in .env"
echo "  2. Start all services: cd docker && docker compose up"
echo "  3. Access the web app: http://localhost:3000"
echo "  4. Access the API: http://localhost:8000"
echo ""
echo "Useful commands:"
echo "  - View logs: docker compose logs -f"
echo "  - Stop services: docker compose down"
echo "  - Admin tools: docker compose --profile tools up"
echo ""
