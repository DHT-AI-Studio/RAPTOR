#!/bin/sh
set -e

REALM="master"

BOOTSTRAP_USER="$KC_BOOTSTRAP_ADMIN_USERNAME"
BOOTSTRAP_PASSWORD="$KC_BOOTSTRAP_ADMIN_PASSWORD"

PERMANENT_MASTER_ADMIN_USER="$PERMANENT_MASTER_ADMIN_USER"
PERMANENT_ADMIN_PASSWORD="$PERMANENT_ADMIN_PASSWORD"

KC_URL="${KEYCLOAK_URL:-http://keycloak:8080}"
KCADM="/opt/keycloak/bin/kcadm.sh"

# ------------------------
# 若 permanent admin 已存在則直接跳過
# ------------------------
echo "Checking if permanent admin already exists..."
if $KCADM config credentials \
    --server "$KC_URL" \
    --realm "$REALM" \
    --user "$PERMANENT_MASTER_ADMIN_USER" \
    --password "$PERMANENT_ADMIN_PASSWORD" \
    >/dev/null 2>&1; then
  echo "✅ Permanent admin already initialized. Skipping setup."
  exit 0
fi

# ------------------------
# 等待 Keycloak 並嘗試登入 bootstrap admin
# ------------------------
echo "Waiting for Keycloak and logging in as bootstrap admin..."

TRIES=0
MAX_TRIES=20
SLEEP_SECONDS=3

until $KCADM config credentials \
    --server "$KC_URL" \
    --realm "$REALM" \
    --user "$BOOTSTRAP_USER" \
    --password "$BOOTSTRAP_PASSWORD" \
    >/dev/null 2>&1
do
  TRIES=$((TRIES+1))
  if [ "$TRIES" -ge "$MAX_TRIES" ]; then
    echo
    echo "❌ Failed to login after $MAX_TRIES attempts."
    exit 1
  fi
  echo -n "."
  sleep "$SLEEP_SECONDS"
done

echo
echo "✅ Successfully logged in as bootstrap admin."

# ------------------------
# 建立永久 admin（如果不存在）
# ------------------------
EXISTS=$(
  $KCADM get users -r "$REALM" \
    -q username="$PERMANENT_MASTER_ADMIN_USER" \
    --fields username \
    --format csv | tail -n 1
)

if [ "$EXISTS" != "$PERMANENT_MASTER_ADMIN_USER" ]; then
  echo "Creating permanent admin user..."
  $KCADM create users -r "$REALM" \
    -s username="$PERMANENT_MASTER_ADMIN_USER" \
    -s enabled=true
else
  echo "Permanent admin already exists."
fi

# ------------------------
# 設定永久 admin 密碼
# ------------------------
echo "Setting password for permanent admin..."
$KCADM set-password -r "$REALM" \
  --username "$PERMANENT_MASTER_ADMIN_USER" \
  --new-password "$PERMANENT_ADMIN_PASSWORD" \
  --temporary=false

# ------------------------
# 指派 admin realm role
# ------------------------
echo "Assigning admin role..."
$KCADM add-roles -r "$REALM" \
  --uusername "$PERMANENT_MASTER_ADMIN_USER" \
  --rolename admin

# ------------------------
# 刪除 bootstrap admin
# ------------------------
echo "Deleting bootstrap admin user..."
BOOTSTRAP_USER_ID=$(
  $KCADM get users -r "$REALM" \
    -q username="$BOOTSTRAP_USER" \
    --fields id \
    --format csv | tail -n 1 | tr -d '"'
)

if [ -n "$BOOTSTRAP_USER_ID" ]; then
  $KCADM delete users/"$BOOTSTRAP_USER_ID" -r "$REALM"
  echo "Bootstrap admin '$BOOTSTRAP_USER' deleted."
else
  echo "Bootstrap admin '$BOOTSTRAP_USER' not found, skipping."
fi

echo "✅ Permanent admin setup complete."