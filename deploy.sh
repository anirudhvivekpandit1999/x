```bash
#!/usr/bin/env bash
set -euo pipefail

# Copy & paste this deploy.sh to /home/ec2-user/x/deploy.sh, then run:
# chmod +x /home/ec2-user/x/deploy.sh

# 1. Navigate to app directory
cd /home/ec2-user/x

# 2. Fetch and reset to main for a clean sync
git fetch origin main
git reset --hard origin/main

# 3. Activate virtualenv if it exists
if [ -f venv/bin/activate ]; then
  source venv/bin/activate
fi

# 4. Install dependencies
pip install -r requirements.txt

# 5. Restart app via PM2
pm2 restart all

# 6. Return to original directory
cd -

echo "✅ Deploy complete: $(date +'%Y-%m-%d %H:%M:%S')"
```
