#!/bin/bash

# FastAPI Gateway Docker 部署腳本
# 用途: 簡化 Docker Compose 操作

set -e

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 項目名稱
PROJECT_NAME="vie-api-gateway"
COMPOSE_FILE="docker-compose.yml"

# 打印帶顏色的消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 顯示幫助
show_help() {
    cat << EOF
FastAPI Gateway Docker 部署腳本

用法: ./deploy.sh [命令] [選項]

命令:
  up              啟動所有服務 (後台運行)
  down            停止所有服務
  restart         重新啟動所有服務
  build           構建 Docker 映像
  rebuild         重新構建映像（不使用快取）
  logs            查看所有服務日誌
  logs-gateway    查看 API Gateway 日誌
  logs-postgres   查看 PostgreSQL 日誌
  logs-pgadmin    查看 PgAdmin 日誌
  ps              查看容器狀態
  shell           進入 API Gateway 容器
  db-shell        進入 PostgreSQL 容器
  db-backup       備份數據庫
  db-restore      恢復數據庫
  health          檢查服務健康狀態
  clean           清除所有容器和卷（危險！）
  help            顯示此幫助信息

選項:
  -f, --follow    用於 logs 命令：實時跟蹤日誌

示例:
  ./deploy.sh up              # 啟動所有服務
  ./deploy.sh logs -f         # 實時跟蹤所有日誌
  ./deploy.sh logs-gateway -f # 跟蹤 Gateway 日誌
  ./deploy.sh shell           # 進入 Gateway 容器

EOF
}

# 檢查 Docker 和 Docker Compose
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安裝"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安裝"
        exit 1
    fi
    
    print_success "Docker 和 Docker Compose 已安裝"
}

# 啟動服務
start_services() {
    print_info "啟動所有服務..."
    docker-compose -f $COMPOSE_FILE up -d
    print_success "服務已啟動"
    
    print_info "等待服務初始化..."
    sleep 5
    
    show_status
}

# 停止服務
stop_services() {
    print_info "停止所有服務..."
    docker-compose -f $COMPOSE_FILE stop
    print_success "服務已停止"
}

# 重新啟動服務
restart_services() {
    print_info "重新啟動所有服務..."
    docker-compose -f $COMPOSE_FILE restart
    print_success "服務已重新啟動"
    
    print_info "等待服務初始化..."
    sleep 5
    
    show_status
}

# 構建映像
build_image() {
    print_info "構建 Docker 映像..."
    docker-compose -f $COMPOSE_FILE build
    print_success "映像構建完成"
}

# 重新構建映像
rebuild_image() {
    print_info "重新構建 Docker 映像（不使用快取）..."
    docker-compose -f $COMPOSE_FILE build --no-cache
    print_success "映像構建完成"
}

# 查看日誌
view_logs() {
    local follow_flag=""
    if [[ "$2" == "-f" ]] || [[ "$2" == "--follow" ]]; then
        follow_flag="-f"
    fi
    
    print_info "顯示日誌..."
    docker-compose -f $COMPOSE_FILE logs $follow_flag
}

# 查看 Gateway 日誌
view_gateway_logs() {
    local follow_flag=""
    if [[ "$2" == "-f" ]] || [[ "$2" == "--follow" ]]; then
        follow_flag="-f"
    fi
    
    print_info "顯示 API Gateway 日誌..."
    docker-compose -f $COMPOSE_FILE logs $follow_flag api-gateway
}

# 查看 PostgreSQL 日誌
view_postgres_logs() {
    local follow_flag=""
    if [[ "$2" == "-f" ]] || [[ "$2" == "--follow" ]]; then
        follow_flag="-f"
    fi
    
    print_info "顯示 PostgreSQL 日誌..."
    docker-compose -f $COMPOSE_FILE logs $follow_flag postgres
}

# 查看 PgAdmin 日誌
view_pgadmin_logs() {
    local follow_flag=""
    if [[ "$2" == "-f" ]] || [[ "$2" == "--follow" ]]; then
        follow_flag="-f"
    fi
    
    print_info "顯示 PgAdmin 日誌..."
    docker-compose -f $COMPOSE_FILE logs $follow_flag pgadmin
}

# 查看容器狀態
show_status() {
    print_info "容器狀態:"
    docker-compose -f $COMPOSE_FILE ps
}

# 進入 Gateway 容器
enter_gateway_shell() {
    print_info "進入 API Gateway 容器..."
    docker-compose -f $COMPOSE_FILE exec api-gateway bash
}

# 進入 PostgreSQL 容器
enter_db_shell() {
    print_info "進入 PostgreSQL 容器..."
    docker-compose -f $COMPOSE_FILE exec postgres psql -U admin -d mydb
}

# 備份數據庫
backup_database() {
    local backup_file="backup_$(date +%Y%m%d_%H%M%S).sql"
    print_info "備份數據庫到 $backup_file..."
    docker-compose -f $COMPOSE_FILE exec postgres pg_dump -U admin mydb > "$backup_file"
    print_success "數據庫已備份到 $backup_file"
}

# 恢復數據庫
restore_database() {
    if [ -z "$2" ]; then
        print_error "請提供備份文件路徑"
        echo "用法: ./deploy.sh db-restore <backup_file>"
        exit 1
    fi
    
    local backup_file="$2"
    
    if [ ! -f "$backup_file" ]; then
        print_error "備份文件不存在: $backup_file"
        exit 1
    fi
    
    print_warning "此操作將覆蓋現有數據庫。繼續？(y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        print_info "恢復已取消"
        exit 0
    fi
    
    print_info "恢復數據庫..."
    docker-compose -f $COMPOSE_FILE exec -T postgres psql -U admin mydb < "$backup_file"
    print_success "數據庫已恢復"
}

# 檢查服務健康狀態
check_health() {
    print_info "檢查服務健康狀態..."
    
    # 檢查 API Gateway
    print_info "檢查 API Gateway..."
    if curl -s http://localhost:8012/health > /dev/null; then
        print_success "API Gateway 健康"
    else
        print_error "API Gateway 無法訪問"
    fi
    
    # 檢查 PostgreSQL
    print_info "檢查 PostgreSQL..."
    if docker-compose -f $COMPOSE_FILE exec -T postgres pg_isready -U admin -d mydb > /dev/null 2>&1; then
        print_success "PostgreSQL 健康"
    else
        print_error "PostgreSQL 無法訪問"
    fi
    
    # 檢查 PgAdmin
    print_info "檢查 PgAdmin..."
    if curl -s http://localhost:5480 > /dev/null; then
        print_success "PgAdmin 健康"
    else
        print_error "PgAdmin 無法訪問"
    fi
}

# 清理所有容器和卷
clean_all() {
    print_warning "此操作將刪除所有容器、卷和數據。此操作不可逆！"
    print_warning "繼續？(y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        print_info "清理已取消"
        exit 0
    fi
    
    print_info "正在清理..."
    docker-compose -f $COMPOSE_FILE down -v
    print_success "清理完成"
}

# 主入口
main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    local command=$1
    
    case $command in
        up)
            check_docker
            start_services
            ;;
        down)
            check_docker
            stop_services
            ;;
        restart)
            check_docker
            restart_services
            ;;
        build)
            check_docker
            build_image
            ;;
        rebuild)
            check_docker
            rebuild_image
            ;;
        logs)
            check_docker
            view_logs "$@"
            ;;
        logs-gateway)
            check_docker
            view_gateway_logs "$@"
            ;;
        logs-postgres)
            check_docker
            view_postgres_logs "$@"
            ;;
        logs-pgadmin)
            check_docker
            view_pgadmin_logs "$@"
            ;;
        ps)
            check_docker
            show_status
            ;;
        shell)
            check_docker
            enter_gateway_shell
            ;;
        db-shell)
            check_docker
            enter_db_shell
            ;;
        db-backup)
            check_docker
            backup_database
            ;;
        db-restore)
            check_docker
            restore_database "$@"
            ;;
        health)
            check_docker
            check_health
            ;;
        clean)
            check_docker
            clean_all
            ;;
        help)
            show_help
            ;;
        *)
            print_error "未知命令: $command"
            show_help
            exit 1
            ;;
    esac
}

# 執行主函數
main "$@"
