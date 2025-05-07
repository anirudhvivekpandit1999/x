#!/usr/bin/env bash
set -euo pipefail

# Copy & paste this deploy.sh to /home/ec2-user/x/deploy.sh, then run:
# chmod +x /home/ec2-user/x/deploy.sh

# 1. Navigate to app directory
cd /home/ec2-user/x

# 2. Pull latest changes from main
git pull origin main

# 3. (Optional) Activate virtualenv if you have one
if [ -f venv/bin/activate ]; then
source venv/bin/activate
fi

# 4. (Optional) Install new dependencies
pip install -r requirements.txt

# 5. Restart your app via PM2
pm2 restart all

# 6. Return to original directory (keeps history clean)
cd -

echo "✅ Deploy complete: $(date +'%Y-%m-%d %H:%M:%S')"