# Frontend Deployment Guide

## Quick Restart After Rebuild

After rebuilding the frontend, you need to restart the web server that serves it. The method depends on how your frontend is deployed:

### Option 1: Using the Rebuild Script (Recommended)

```bash
./rebuild_frontend.sh
```

This script will:
1. Build the frontend (`npm run build`)
2. Automatically detect and reload the web server (nginx, PM2, or systemd)

### Option 2: Manual Steps

#### If Using Nginx

1. **Rebuild the frontend:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Reload nginx:**
   ```bash
   sudo systemctl reload nginx
   # OR
   sudo service nginx reload
   ```

3. **Verify nginx is serving the new build:**
   ```bash
   sudo nginx -t  # Test configuration
   sudo systemctl status nginx  # Check status
   ```

#### If Using PM2 (with serve or similar)

1. **Rebuild the frontend:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Restart PM2 process:**
   ```bash
   pm2 restart frontend
   # OR if using a different name
   pm2 restart serve
   pm2 restart static-server
   ```

3. **Check PM2 status:**
   ```bash
   pm2 list
   pm2 logs frontend
   ```

#### If Using Systemd Service

1. **Rebuild the frontend:**
   ```bash
   cd frontend
   npm run build
   ```

2. **Restart the service:**
   ```bash
   sudo systemctl restart rag-anything-frontend
   # OR whatever your service name is
   ```

3. **Check service status:**
   ```bash
   sudo systemctl status rag-anything-frontend
   ```

## Common Deployment Setups

### Nginx Configuration

If nginx is serving the frontend, your config might look like:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    root /path/to/rag-anything/frontend/dist;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # API proxy
    location /api {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

After rebuilding, just reload nginx:
```bash
sudo systemctl reload nginx
```

### PM2 with serve

If using PM2 with a static file server:

```bash
# Install serve globally
npm install -g serve

# Start with PM2
pm2 serve frontend/dist 3000 --name frontend --spa

# After rebuild, just restart
pm2 restart frontend
```

### Development Server (Not Recommended for Production)

If running the Vite dev server directly:

```bash
cd frontend
npm run dev
```

**Note:** This is not recommended for production. Use a production build instead.

## Troubleshooting

### Changes Not Appearing

1. **Clear browser cache:**
   - Hard refresh: `Ctrl+Shift+R` (Linux/Windows) or `Cmd+Shift+R` (Mac)
   - Or clear browser cache completely

2. **Check if new files were created:**
   ```bash
   ls -la frontend/dist/
   # Check timestamps to verify new build
   ```

3. **Verify web server is serving from correct directory:**
   ```bash
   # For nginx
   sudo nginx -T | grep root
   
   # Check what's actually being served
   curl -I http://your-server/
   ```

### Build Errors

If the build fails:

```bash
cd frontend
npm install  # Reinstall dependencies
npm run build  # Try building again
```

Check for errors in the build output.

### 404 Errors After Deploy

If you get 404 errors for routes:

- **For nginx:** Make sure you have the `try_files` directive (see nginx config above)
- **For PM2 serve:** Use the `--spa` flag to enable SPA mode
- **For other servers:** Configure them to serve `index.html` for all routes

## Production Best Practices

1. **Always use production build:**
   ```bash
   npm run build
   ```

2. **Don't run dev server in production:**
   - Use nginx, PM2 with serve, or another production web server
   - Dev server (`npm run dev`) is for development only

3. **Set up proper caching:**
   - Cache static assets (JS, CSS) with long expiration
   - Don't cache `index.html` (or use short cache with versioning)

4. **Use HTTPS in production:**
   - Configure SSL certificates
   - Redirect HTTP to HTTPS

5. **Monitor and log:**
   - Check web server logs regularly
   - Set up error monitoring

## Quick Reference

```bash
# Full rebuild and restart
./rebuild_frontend.sh

# Or manual:
cd frontend && npm run build && cd ..
sudo systemctl reload nginx  # If using nginx
pm2 restart frontend         # If using PM2
```

