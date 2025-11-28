# RAG Chat Frontend - React Web App

This is a React-based web application for the RAG (Retrieval-Augmented Generation) chat system.

## Requirements

- Node.js 16+ and npm
- Backend server running (default: `http://127.0.0.1:8000`)

## Installation

```bash
npm install
```

## Configuration

The application can be configured using environment variables. See [ENV_CONFIGURATION.md](./ENV_CONFIGURATION.md) for detailed instructions.

### Quick Setup

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` to customize settings (optional):
```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
VITE_APP_TITLE=RAG Chat Aplikace
```

3. Restart the dev server to apply changes

## Running the Application

### Development Mode

```bash
npm run dev
```

The application will start on `http://localhost:3000`

### Production Build

```bash
npm run build
```

The built files will be in the `dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Features

- **Chat Interface**: Chat with your documents using AI
- **Document Management**: Upload and manage documents (PDF, DOCX, PPTX, XLSX, HTML, images, audio, TXT, MD)
- **Real-time Processing**: Track document processing progress
- **Drag & Drop**: Easy file upload via drag and drop

## Backend API

The frontend connects to the backend API (default: `http://127.0.0.1:8000`). Make sure the backend is running before starting the frontend.

You can configure the backend URL using the `VITE_API_BASE_URL` environment variable. See [ENV_CONFIGURATION.md](./ENV_CONFIGURATION.md) for details.

Key endpoints:
- `GET /health` - Check backend health
- `GET /chat/messages` - Get chat history
- `POST /chat/send` - Send a message
- `POST /chat/clear` - Clear chat history
- `GET /documents` - List documents
- `DELETE /documents/{id}` - Delete a document
- `POST /upload` - Upload a document
- `GET /{filename}/status` - Check processing status

## Supported File Types

- Documents: PDF, DOCX, PPTX, XLSX, TXT, MD
- Images: PNG, JPG, JPEG, TIFF, BMP, GIF
- Audio: WAV, MP3, VTT
- Web: HTML, HTM

## Converting from Tauri Desktop App

This application was converted from a Tauri desktop app to a plain React web app. The main changes were:

1. Removed `@tauri-apps/api` and `@tauri-apps/cli` dependencies
2. Replaced `invoke()` calls with `fetch()` API calls to the backend
3. Replaced Tauri file dialog with HTML file input
4. Updated Vite configuration for web deployment

## Project Structure

```
frontend/
├── src/
│   ├── App.tsx              # Main application component
│   ├── config.ts            # Configuration and environment variables
│   ├── main.tsx             # Application entry point
│   └── index.css            # Global styles
├── .env                     # Local environment config (not in git)
├── .env.example             # Environment config template
├── env.d.ts                 # TypeScript env variable definitions
├── ENV_CONFIGURATION.md     # Detailed configuration guide
└── package.json             # Dependencies and scripts
```

## Development Notes

- **Styling:** Tailwind CSS
- **Icons:** lucide-react
- **State Management:** React hooks
- **File Uploads:** HTML File API
- **Configuration:** Environment variables via Vite
- **API Calls:** Fetch API

