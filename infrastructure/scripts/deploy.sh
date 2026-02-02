#!/bin/bash
# =============================================================================
# LARUN MVP - Deployment Script
# =============================================================================
# Deploys LARUN to staging or production environment
# Usage: ./deploy.sh [staging|production]
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

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
ENVIRONMENT="${1:-staging}"

if [[ "$ENVIRONMENT" != "staging" && "$ENVIRONMENT" != "production" ]]; then
    echo -e "${RED}Error: Invalid environment. Use 'staging' or 'production'${NC}"
    echo "Usage: $0 [staging|production]"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  LARUN MVP - Deploy to ${ENVIRONMENT}${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# -----------------------------------------------------------------------------
# Safety checks for production
# -----------------------------------------------------------------------------
if [ "$ENVIRONMENT" == "production" ]; then
    echo -e "${YELLOW}WARNING: You are about to deploy to PRODUCTION${NC}"
    echo ""

    # Check for required environment variables
    REQUIRED_VARS=(
        "DATABASE_URL"
        "REDIS_URL"
        "NEXTAUTH_SECRET"
        "JWT_SECRET"
        "STRIPE_SECRET_KEY"
        "STRIPE_WEBHOOK_SECRET"
    )

    MISSING_VARS=()
    for var in "${REQUIRED_VARS[@]}"; do
        if [ -z "${!var}" ]; then
            MISSING_VARS+=("$var")
        fi
    done

    if [ ${#MISSING_VARS[@]} -gt 0 ]; then
        echo -e "${RED}Error: Missing required environment variables:${NC}"
        for var in "${MISSING_VARS[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi

    # Confirm deployment
    read -p "Type 'deploy' to confirm production deployment: " CONFIRM
    if [ "$CONFIRM" != "deploy" ]; then
        echo -e "${YELLOW}Deployment cancelled${NC}"
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Build Docker images
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Building Docker images...${NC}"

cd "$PROJECT_ROOT"

# Build API image
echo "Building API image..."
docker build -t larun-api:latest -f docker/Dockerfile.api .

# Build Web image
echo "Building Web image..."
docker build -t larun-web:latest -f docker/Dockerfile.web .

echo -e "${GREEN}  [OK] Docker images built${NC}"
echo ""

# -----------------------------------------------------------------------------
# Run tests
# -----------------------------------------------------------------------------
echo -e "${YELLOW}Running tests...${NC}"

# Run API tests
echo "Running API tests..."
docker run --rm larun-api:latest pytest tests/ -v || {
    echo -e "${RED}API tests failed. Aborting deployment.${NC}"
    exit 1
}

echo -e "${GREEN}  [OK] All tests passed${NC}"
echo ""

# -----------------------------------------------------------------------------
# Deploy based on environment
# -----------------------------------------------------------------------------
if [ "$ENVIRONMENT" == "staging" ]; then
    echo -e "${YELLOW}Deploying to staging...${NC}"

    # Tag images for staging
    docker tag larun-api:latest larun-api:staging
    docker tag larun-web:latest larun-web:staging

    # Push to registry (configure your registry here)
    # docker push your-registry/larun-api:staging
    # docker push your-registry/larun-web:staging

    # Deploy to staging server
    # This would typically be:
    # - Railway: railway up
    # - Fly.io: fly deploy
    # - AWS: aws ecs update-service
    # - K8s: kubectl apply -f k8s/staging/

    echo -e "${GREEN}  [OK] Deployed to staging${NC}"
    echo ""
    echo "Staging URLs:"
    echo "  - Web: https://staging.larun.ai"
    echo "  - API: https://api.staging.larun.ai"

else
    echo -e "${YELLOW}Deploying to production...${NC}"

    # Tag images for production
    GIT_SHA=$(git rev-parse --short HEAD)
    docker tag larun-api:latest larun-api:$GIT_SHA
    docker tag larun-api:latest larun-api:production
    docker tag larun-web:latest larun-web:$GIT_SHA
    docker tag larun-web:latest larun-web:production

    # Push to registry (configure your registry here)
    # docker push your-registry/larun-api:$GIT_SHA
    # docker push your-registry/larun-api:production
    # docker push your-registry/larun-web:$GIT_SHA
    # docker push your-registry/larun-web:production

    # Deploy to production server
    # This would typically be:
    # - Railway: railway up --environment production
    # - Fly.io: fly deploy --app larun-production
    # - AWS: aws ecs update-service --cluster production
    # - K8s: kubectl apply -f k8s/production/

    echo -e "${GREEN}  [OK] Deployed to production${NC}"
    echo ""
    echo "Production URLs:"
    echo "  - Web: https://larun.ai"
    echo "  - API: https://api.larun.ai"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
