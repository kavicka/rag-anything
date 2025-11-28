#!/bin/bash

# 🚀 Quick Start Script for RAG Chat App
# This script starts both backend and frontend

echo "🚀 Starting RAG Chat App..."
echo "=========================="

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "backend" ]; then
    echo "❌ Please run this script from the RAG project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run: python3 -m venv .venv"
    exit 1
fi

# Function to start backend
start_backend() {
    echo "🐍 Starting Python Backend..."
    cd backend
    ../.venv/bin/python main.py &
    BACKEND_PID=$!
    echo "Backend started with PID: $BACKEND_PID"
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "⚛️ Starting Tauri Frontend..."
    npm run tauri:dev &
    FRONTEND_PID=$!
    echo "Frontend started with PID: $FRONTEND_PID"
}

# Function to cleanup on exit
cleanup() {
    echo "🛑 Shutting down..."
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start services
start_backend
sleep 3  # Give backend time to start
start_frontend

echo ""
echo "🎉 RAG Chat App is starting!"
echo "📚 Backend API: http://127.0.0.1:8000"
echo "📖 API Docs: http://127.0.0.1:8000/docs"
echo "🖥️  Frontend: Tauri desktop app (opening...)"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
wait
