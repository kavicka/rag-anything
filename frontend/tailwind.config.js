/** @type {import('tailwindcss').Config} */
export default {
    content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
    theme: {
        extend: {
            colors: {
                'chat-bg': '#f8fafc',
                'sidebar-bg': '#1e293b',
                'message-user': '#3b82f6',
                'message-assistant': '#e5e7eb',
            },
        },
    },
    plugins: [],
}
