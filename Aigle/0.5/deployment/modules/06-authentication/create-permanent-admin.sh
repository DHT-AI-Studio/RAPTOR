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
# SMTP 設定 helper（需在 kcadm session 已認證後呼叫）
# ------------------------
configure_smtp() {
  if [ -n "$SMTP_HOST" ]; then
    echo "Configuring SMTP for dhtsolution realm..."
    $KCADM update realms/dhtsolution \
      -s "smtpServer.host=$SMTP_HOST" \
      -s "smtpServer.port=$SMTP_PORT" \
      -s "smtpServer.from=$SMTP_FROM" \
      -s "smtpServer.fromDisplayName=$SMTP_FROM_DISPLAY_NAME" \
      -s "smtpServer.auth=$SMTP_AUTH" \
      -s "smtpServer.user=$SMTP_USER" \
      -s "smtpServer.password=$SMTP_PASSWORD" \
      -s "smtpServer.ssl=$SMTP_SSL" \
      -s "smtpServer.starttls=$SMTP_STARTTLS"
    echo "✅ SMTP configured."
  else
    echo "⚠️ SMTP_HOST not set, skipping SMTP configuration."
  fi
}

# ------------------------
# raptor client secret helper（需在 kcadm session 已認證後呼叫）——
# 讓運行中的 Keycloak 的 raptor client secret 跟 .env 的 RAPTOR_CLIENT_SECRET
# 保持同步，而不是信任 realm-import JSON 裡那個只在第一次 import 當下
# 用得到的初始值。
# ------------------------
configure_raptor_client_secret() {
  if [ -n "$RAPTOR_CLIENT_SECRET" ]; then
    echo "Rotating raptor client secret..."
    CLIENT_UUID=$(
      $KCADM get clients -r dhtsolution \
        -q clientId=raptor \
        --fields id \
        --format csv | tail -n 1 | tr -d '"'
    )
    if [ -n "$CLIENT_UUID" ]; then
      $KCADM update clients/"$CLIENT_UUID" -r dhtsolution \
        -s secret="$RAPTOR_CLIENT_SECRET"
      echo "✅ raptor client secret set from RAPTOR_CLIENT_SECRET."
    else
      echo "⚠️ raptor client not found, skipping secret rotation."
    fi
  else
    echo "⚠️ RAPTOR_CLIENT_SECRET not set, skipping secret rotation."
  fi
}

# ------------------------
# 若 permanent admin 已存在則設定 SMTP 後離開
# ------------------------
echo "Checking if permanent admin already exists..."
if $KCADM config credentials \
    --server "$KC_URL" \
    --realm "$REALM" \
    --user "$PERMANENT_MASTER_ADMIN_USER" \
    --password "$PERMANENT_ADMIN_PASSWORD" \
    >/dev/null 2>&1; then
  echo "✅ Permanent admin already initialized."
  configure_raptor_client_secret
  configure_smtp
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
  printf '.'
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
# 切換成永久 admin 的 session——下面要刪除 bootstrap admin 這個使用者，
# Keycloak 會讓那個使用者自己當下的 token 失效，kcadm 的 session 就跟著
# 斷掉。永久 admin 剛剛才建立、才指派角色，用它重新登入可以確保後面
# configure_raptor_client_secret()/configure_smtp() 用的是不會被這次
# 刪除動作波及的 session。
# ------------------------
echo "Switching kcadm session to permanent admin before deleting bootstrap admin..."
$KCADM config credentials \
  --server "$KC_URL" \
  --realm "$REALM" \
  --user "$PERMANENT_MASTER_ADMIN_USER" \
  --password "$PERMANENT_ADMIN_PASSWORD"

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

# ------------------------
# 設定 raptor client secret、dhtsolution realm SMTP
# ------------------------
configure_raptor_client_secret
configure_smtp

echo "✅ Permanent admin setup complete."