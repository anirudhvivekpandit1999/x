set -euo pipefail
cd /home/ec2-user/x
git pull origin main
pm2 restart all
cd -
echo "✅ Deploy complete: $(date +'%Y-%m-%d %H:%M:%S')"
