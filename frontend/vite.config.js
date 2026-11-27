import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
    plugins: [react()],

    // Tell Vite to use main.html as the entry point
    // for both dev server and production build
    build: {
        rollupOptions: {
            input: resolve(__dirname, 'main.html'),
        },
    },

    // Dev server configuration
    server: {
        // Serve main.html when visiting /
        open: '/main.html',

        // Proxy API requests to Flask during development
        proxy: {
            '/detect': 'http://localhost:5000',
            '/health': 'http://localhost:5000',
        },
    },
})