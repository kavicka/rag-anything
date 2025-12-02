#!/bin/bash
# Rebuild and restart frontend on server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="${SCRIPT_DIR}/frontend"
DIST_DIR="${FRONTEND_DIR}/dist"

echo "=========================================="
echo "Rebuilding Frontend"
echo "=========================================="

# Navigate to frontend directory
cd "${FRONTEND_DIR}"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build the frontend
echo "Building frontend..."
npm run build

if [ ! -d "dist" ]; then
    echo "ERROR: Build failed - dist directory not created"
    exit 1
fi

echo "✅ Frontend built successfully"
echo "Build output: ${DIST_DIR}"

# Check if nginx is serving the frontend
if command -v nginx &> /dev/null; then
    echo ""
    echo "Checking nginx configuration..."
    
    # Try to reload nginx if it's running
    if systemctl is-active --quiet nginx 2>/dev/null || service nginx status &>/dev/null; then
        echo "Reloading nginx..."
        sudo systemctl reload nginx 2>/dev/null || sudo service nginx reload 2>/dev/null || {
            echo "⚠️  Could not reload nginx automatically"
            echo "   Please run manually: sudo systemctl reload nginx"
        }
        echo "✅ Nginx reloaded"
    else
        echo "⚠️  Nginx is not running"
    fi
fi

# Check if PM2 is serving the frontend (via serve or similar)
if command -v pm2 &> /dev/null; then
    PM2_FRONTEND=$(pm2 list | grep -i "frontend\|serve\|static" || true)
    if [ ! -z "$PM2_FRONTEND" ]; then
        echo ""
        echo "Restarting PM2 frontend process..."
        pm2 restart frontend 2>/dev/null || pm2 restart serve 2>/dev/null || {
            echo "⚠️  Could not restart PM2 frontend process"
            echo "   Check PM2 processes: pm2 list"
        }
    fi
fi

# Check if there's a systemd service for the frontend
if systemctl list-units --type=service | grep -q "rag-anything-frontend"; then
    echo ""
    echo "Restarting frontend systemd service..."
    sudo systemctl restart rag-anything-frontend
    echo "✅ Frontend service restarted"
fi

echo ""
echo "=========================================="
echo "Frontend Rebuild Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If using nginx, verify it's serving from: ${DIST_DIR}"
echo "2. Clear browser cache if changes don't appear"
echo "3. Check frontend logs if there are issues"
echo ""
echo "To verify nginx config:"
echo "  sudo nginx -t"
echo "  sudo systemctl status nginx"
echo ""

