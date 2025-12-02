// PM2 Ecosystem Configuration for RAG-Anything Backend
// Alternative to systemd for process management

module.exports = {
  apps: [{
    name: 'rag-anything-backend',
    script: 'api_server.py',
    interpreter: '/opt/rag-anything/venv/bin/python',
    cwd: '/opt/rag-anything',
    instances: 1,
    exec_mode: 'fork',
    autorestart: true,
    watch: false,
    max_memory_restart: '2G',
    env: {
      NODE_ENV: 'production',
      PYTHONUNBUFFERED: '1'
    },
    error_file: '/opt/rag-anything/logs/pm2-error.log',
    out_file: '/opt/rag-anything/logs/pm2-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    time: true
  }]
};

