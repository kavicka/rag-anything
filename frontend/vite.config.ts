import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        host: '0.0.0.0', // Listen on all interfaces for server deployment
        port: 5173, // Changed from 3000 to avoid conflict with other service
        strictPort: false,
    },
})
