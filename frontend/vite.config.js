import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        proxy: {
            '/api': {
                target: 'http://localhost:12000',
                changeOrigin: true,
            },
            '/execute-pipeline-from-file': { target: 'http://localhost:12000', changeOrigin: true },
            '/execute-pipeline-from-db': { target: 'http://localhost:12000', changeOrigin: true },
            '/create-pipeline-tracker': { target: 'http://localhost:12000', changeOrigin: true },
            '/pipeline-progress-stream': { target: 'http://localhost:12000', changeOrigin: true, ws: true },
            '/pipeline-progress': { target: 'http://localhost:12000', changeOrigin: true },
            '/configure-db-source-pipeline': { target: 'http://localhost:12000', changeOrigin: true },
            '/configure-preembedded-pipeline': { target: 'http://localhost:12000', changeOrigin: true },
            '/execute-preembedded-pipeline': { target: 'http://localhost:12000', changeOrigin: true },
            '/unified-search': { target: 'http://localhost:12000', changeOrigin: true },
            '/generate-summary': { target: 'http://localhost:12000', changeOrigin: true },
            '/generate-summary-stream': { target: 'http://localhost:12000', changeOrigin: true },
            '/export-config': { target: 'http://localhost:12000', changeOrigin: true },
            '/import-config': { target: 'http://localhost:12000', changeOrigin: true },
            '/list-configs': { target: 'http://localhost:12000', changeOrigin: true },
            '/get-source-db-columns': { target: 'http://localhost:12000', changeOrigin: true },
            '/cancel-pipeline': { target: 'http://localhost:12000', changeOrigin: true },
            '/reload-indexes': { target: 'http://localhost:12000', changeOrigin: true },
        }
    }
})
