# src/core/model_manager.py
import os
import shutil
import lakefs
from lakefs.client import Client
from lakefs import exceptions

import mlflow
from mlflow.tracking import MlflowClient
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, snapshot_download
from ollama import Client as OllamaClient
from .config import config
from pathlib import Path
import datetime

from urllib.parse import urlparse
from typing import Optional, Dict, Any, List
from ..inference.vram_estimator import ModelResourceEstimator

class ModelManager:
    """
    負責模型的下載、上傳、註冊和版本管理。
    支援 HuggingFace 和 Ollama 兩種模型來源，並與 lakeFS 和 MLflow 整合。
    """
    def __init__(self):
        self.mlflow_uri = config.get_config("mlflow", "tracking_uri")
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)
        self.lakefs_client = Client(host=config.get_config("lakefs", "endpoint"),
                                            username=config.get_config("lakefs", "access_key"), 
                                            password=config.get_config("lakefs", "secret_key"))
        self.lakefs_models_storage_namespace = config.get_config("lakefs", "model_storage_namespace") # lakeFS 存儲命名空間
        self.models_tmp_root = os.path.abspath(config.get_config("storage", "models_tmp_root"))  # 用於存放從網路上下載至本地的模型
        os.makedirs(self.models_tmp_root, exist_ok=True)
        self.download_from_lakefs_root = os.path.abspath(config.get_config("storage", "models_from_lakefs_root")) # 用於存放從 lakeFS 下載至本地的模型
        os.makedirs(self.download_from_lakefs_root, exist_ok=True)
        self.hf_token = config.get_config("huggingface", "hf_token")
        self.inference_task_rules = config.get_config("inference", "task_to_models")
        self.inference_engines = config.get_config("inference", "engines")

        self.estimator = ModelResourceEstimator()

    def _infer_precision_from_quantization(self, quantization_level: str) -> str:
        """
        根據 Ollama 量化級別推斷精度類型。
        
        Args:
            quantization_level (str): Ollama 量化級別，如 "Q4_0", "Q8_0", "F16" 等
            
        Returns:
            str: 對應的精度類型 ("int8", "fp16", "fp32")
        """
        if not quantization_level:
            return "fp16"  # 預設值
            
        quant = quantization_level.upper()
        
        # 根據 Ollama 量化級別映射到精度
        if quant.startswith('Q4') or quant.startswith('Q5') or quant.startswith('Q6'):
            return "int8"  # 4-6 bit 量化近似為 int8
        elif quant.startswith('Q8'):
            return "int8"  # 8 bit 量化
        elif quant.startswith('F16') or quant.startswith('FP16'):
            return "fp16"  # 16 bit 浮點
        elif quant.startswith('F32') or quant.startswith('FP32'):
            return "fp32"  # 32 bit 浮點
        else:
            print(f"Warning: Unknown quantization level '{quantization_level}', defaulting to fp16")
            return "fp16"  # 未知類型預設為 fp16

    def download_model(self, 
                       model_source: str, 
                       model_name: str, 
                       destination_path: Optional[str] = None) -> str:
        """
        從 HuggingFace 或 Ollama 下載模型到指定路徑
        Args:
            model_source: "huggingface" 或 "ollama"
            model_name: 網路上的模型名稱或ID（如 "google/gemma-3-270m-it" 或 "llama2"）
            destination_path: 如果指定，下載到這個路徑；否則使用默認路徑(self.models_tmp_root)
        Returns:
            下載後模型的本地路徑或錯誤信息
        """
        if destination_path is None:
            model_name_replace = model_name.replace("/", "_")  # 將 '/' 替換為 '_' 以避免路徑問題
            destination_path = os.path.join(self.models_tmp_root, model_name_replace)
        else:
            destination_path = os.path.join(os.path.abspath(destination_path), model_name_replace)

        os.makedirs(destination_path, exist_ok=True)
        print(f"Downloading model '{model_name_replace}' to '{destination_path}'...")

        if model_source == "huggingface":
            # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, token=self.hf_token)
            # model.save_pretrained(destination_path)
            # tokenizer.save_pretrained(destination_path)
            download_path = snapshot_download(repo_id=model_name, 
                                              repo_type="model", 
                                              cache_dir=destination_path, 
                                              local_dir=destination_path, 
                                              token=self.hf_token,)
            print("HuggingFace model downloaded successfully.")
            return download_path
        elif model_source == "ollama":
            # Ollama 的 "下載" 是通過 pull 命令完成的，通常由 Ollama 服務自己管理。
            # 這裡我們模擬一個記錄過程。
            print("Pulling Ollama model...")
            ollama_cli = OllamaClient(host=config.get_config("ollama", "api_base"))
            ollama_cli.pull(model_name)
            
            print(f"Ollama model {model_name} pulled successfully.")
            return f"Ollama model {model_name} pulled successfully."
        
        else:
            try:
                raise ValueError("Unsupported model source. Use 'huggingface' or 'ollama'.")
            except ValueError as ve:
                return f"Error: {ve}"
  
    
    def list_local_models(self, 
                          model_source: Optional[str] = None, 
                          from_lakefs: Optional[bool] = False) -> list:
        """
        從本地文件系統掃描模型
        Args:
            model_source: 目前只可填"ollama"，默認為 None 時掃描所有本地非ollama模型
            from_lakefs: 是否掃描從 lakeFS 下載的模型目錄，默認為 False，設為True 只會返回從 lakeFS 下載下來的非 ollama 模型，不可model_source設為ollama時True
        Returns:
            list: 模型列表
            如果 model_source 為 "ollama"，返回 Ollama 模型列表
            如果 model_source 為 None，返回本地模型列表
            ollama 模型列表格式:
            [
                {'model_name': 'gemma3:270m', 
                'model_size_gb': 0.27,
                'parameter_size': '268.10M',
                'quantization_level': 'Q8_0'
                },
                ...
            ]
            本地模型列表格式:
            [
                {'name': 'apple_FastVLM-0.5B',
                'path': '/opt/home/george/george-test/AiModelLifecycle/VIE01/AiModelLifecycle/tmp/models/apple_FastVLM-0.5B'}, 
                ...
            ]

        """
        if from_lakefs: # 只掃描從 lakeFS 下載的模型
            if model_source == "ollama": # ollama 模型不會從 lakeFS 下載
                try:
                    raise ValueError("from_lakefs cannot be True when model_source is 'ollama'.")
                except ValueError as ve:
                    print(f"Error: {ve}")
                    ve_list = [{"error": str(ve)}]
                    return ve_list
            else:
                local_models_for_inference = []
                for item in os.listdir(self.download_from_lakefs_root):
                    item_path = os.path.join(self.download_from_lakefs_root, item)
                    if os.path.isdir(item_path):
                        local_models_for_inference.append({"name": item, "path": item_path})
                return local_models_for_inference
        else: # 掃描從網路上下載的本地模型
            if model_source == "ollama": # 獲取本地 ollama 模型列表                
                ollama_cli = OllamaClient(host=config.get_config("ollama", "api_base"))
                ollama_list = ollama_cli.list()['models'] # ollama.list() 回傳的是一個 dict，模型列表在 'models' 這個 key 裡
                ollama_models = []
                for model_data in ollama_list:
                    model_name = model_data.model
                    model_size_gb = round(model_data.size / (1024**3), 2) # 將 bytes 轉換為 GB 並四捨五入
                    param_size = model_data.details.parameter_size
                    quant_level = model_data.details.quantization_level
                    ollama_models.append({
                        "model_name": model_name,
                        "model_size_gb": model_size_gb,
                        "parameter_size": param_size,
                        "quantization_level": quant_level
                    })
                return ollama_models
            else: # 獲取本地非 ollama 模型列表
                local_tmp_models = []
                for item in os.listdir(self.models_tmp_root):
                    item_path = os.path.join(self.models_tmp_root, item)
                    if os.path.isdir(item_path):
                        local_tmp_models.append({"name": item, "path": item_path})
                return local_tmp_models

    def _ensure_lakefs_repo(self, repo_name: str, branch_name: Optional[str] = "main"):
        """
        使用 lakeFS API 確保儲存庫以及分支存在，若不存在則創建，已存在則返回現有的儲存庫資訊。
        Args:
            repo_name: 儲存庫名稱
            branch_name: 預設分支名稱，默認為 "main"
        Returns:
            Repository or information about the existing repository or None.
        """
        storage_namespace = self.lakefs_models_storage_namespace + "/" + f"{repo_name}"
        try:

            repo = lakefs.repository(repo_name, client=self.lakefs_client).create(storage_namespace, default_branch="main")
            print("Successfully created lakeFS repository:", repo)
            return repo
        
        except exceptions.ConflictException as e:
            print(f"Repository {repo_name} already exists: {e}")
            for repo in lakefs.repositories(client=self.lakefs_client):
                if repo.id == repo_name:
                    print(f"Repo info: {repo}")
            try:
                branch = lakefs.repository(repo_name, client=self.lakefs_client).branch(branch_name).create(source_reference="main")
                print(f"Branch {branch_name} created: {branch}")
                return branch
            except exceptions.ConflictException as e:
                print(f"Branch {branch_name} already exists: {e}")
                for branch in lakefs.repository(repo_name, client=self.lakefs_client).branches():
                    if branch.id == branch_name:
                        print(f"Branch info: {branch}")
                        return branch
            
                
        except exceptions.NotAuthorizedException as e:
            print(f"Authorization error: {e}")  
            return None        
        except exceptions.BadRequestException as e:
            print(f"Bad request: {e}")
            return None
        except Exception as e:
            print(f"Error creating or accessing repository: {e}")
            return None

    def upload_to_lakefs(self, repo_name: str, local_path: str, branch_name: Optional[str] = "main", ) -> str:
        """
        將本地模型資料夾上傳至 lakeFS 儲存庫的指定分支。
        Args:
            repo_name: lakeFS 儲存庫名稱(需符合格式規範：只允許小寫字母、數字、"-")
            branch_name: 分支名稱，默認為 "main"
            local_path: 本地模型資料夾路徑
        Returns:
            lakefs commit URI 或錯誤信息
        """
        repo = self._ensure_lakefs_repo(repo_name, branch_name)
        if repo is None:
            return f"Failed to ensure lakeFS repository: {repo_name}."
        # 上傳模型    
        try:
            print(f"Uploading model from '{local_path}' to lakeFS repo '{repo_name}' on branch '{branch_name}'...")
            branch_to_upload = lakefs.repository(repo_name, client=self.lakefs_client).branch(branch_name)
            model_source_path = Path(local_path).expanduser().resolve()
            folder_name = model_source_path.name
            all_file = model_source_path.rglob('*')
            for file_path in all_file:
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_source_path)
                    destination_path = f"{folder_name}/{relative_path.as_posix()}"
                    branch_to_upload.object(destination_path).upload(
                        file_path.read_bytes(), 
                        mode="w", 
                        pre_sign=False
                    )
        except Exception as e:
            print(f"Error uploading {repo_name} to lakeFS: {e}")
            return f"Failed to upload {repo_name} to lakeFS."
        # 執行提交
        try:
            commit = branch_to_upload.commit(f"Upload model {folder_name} to {branch_name}")
            print(f"Model uploaded and committed to lakeFS. Commit ID: {commit.id}")

            return f"lakefs://{repo_name}/{commit.id}/"
        except Exception as e:
            print(f"Error committing changes to lakeFS: {e}")
            return "Failed to commit changes to lakeFS."

    def list_lakefs_repos(self) -> list:
        """
        列出所有 lakeFS 儲存庫
        Returns:
            list: 儲存庫資訊列表，格式如下：         
        [{
            "repo_id": "my-repo",
            "repo_creation_date": "2023-10-01 12:00:00",
            "storage_namespace": "s3://my-bucket/my-repo",
            "default_branch": "main",
            "branches": [
                {
                    "branch_id": "main",
                    "latest_commit_id": "abcd1234",
                    "latest_commit_time": "2023-10-10 15:30:00",
                    "committer": "admin",
                    "commit_message": "Initial commit"
                }
            ]
        }]
        """
        repos = []
        try:
            for repo in lakefs.repositories(client=self.lakefs_client):
                # Convert timestamp format
                repo = repo.properties
                branches = []
                for branch in lakefs.repository(repo.id, client=self.lakefs_client).branches():
                    # print(f"Repo: {repo.id}, Branch: {branch.id}, Latest Commit id: {branch.get_commit().id}")
                    creation_date = datetime.datetime.fromtimestamp(branch.get_commit().creation_date).strftime('%Y-%m-%d %H:%M:%S')
                    branches.append({
                        "branch_id": branch.id,
                        "latest_commit_id": branch.get_commit().id,
                        "latest_commit_time": creation_date,
                        # "parents": branch.get_commit().parents,
                        # "meta_range_id": branch.get_commit().meta_range_id,
                        "committer": branch.get_commit().committer,
                        "commit_message": branch.get_commit().message,
                        # "metadata": branch.get_commit().metadata,
                        # "unknown": branch.get_commit().unknown,                       
                    })
                repo_creation_date = datetime.datetime.fromtimestamp(repo.creation_date).strftime('%Y-%m-%d %H:%M:%S')
                repos.append({
                    "repo_id": repo.id, 
                    "repo_creation_date": repo_creation_date,
                    "storage_namespace": repo.storage_namespace, 
                    "default_branch": repo.default_branch,
                    "branches": branches
                })
            return repos
        except Exception as e:
            error_list: list = []
            error_msg: str = f"Error listing lakeFS repositories: {e}"
            print(error_msg)
            error_dict: dict = {"error": error_msg}
            error_list.append(error_dict)
            return error_list

    def download_from_lakefs(self, 
                            repo_name: str, 
                            download_name: Optional[str] = None,
                            commit_id: Optional[str] = None,
                            branch_name: Optional[str] = None,
                            destination_path_root: Optional[str] = None) -> str:
        """
        從 lakeFS 儲存庫的指定分支下載模型到本地，用於推理。
        Args:
            repo_name: lakeFS 儲存庫名稱
            download_name: 下載後的本地資料夾名稱，若為 None 則使用 repo_name
            commit_id: 指定的 commit ID，若為 None 則使用 branch_name(main) 的最新 commit
            branch_name: 分支名稱，默認為 None，通常不需填，因為我們會使用 commit_id 來確保是mlflow所註冊的版本
            destination_path: 本地下載路徑，若為 None 則使用 self.download_from_lakefs_root/repo_name
        Returns:
            str: 本地下載路徑或錯誤信息
        """
        if download_name is None:
            download_name = repo_name
        if destination_path_root is None:
            destination_path_root = self.download_from_lakefs_root
        os.makedirs(destination_path_root, exist_ok=True)

        if commit_id is None:
            # 獲取分支的最新 commit ID
            if branch_name is None: # 默認為 main
                branch_name = "main"
            try:
                commit_id = lakefs.repository(repo_name, client=self.lakefs_client).branch(branch_name).get_commit().id
                print(f"Using latest commit ID from branch '{branch_name}': {commit_id}")
            except Exception as e:
                error_msg = f"Failed to fetch latest commit ID from branch '{branch_name}': {e}"
                print(error_msg)
                return error_msg
        destination_path = os.path.join(destination_path_root, download_name, commit_id)
        if os.path.exists(destination_path) and os.listdir(destination_path): # 目錄存在且非空
            print(f"Model already exists in local cache: {destination_path}")
            return destination_path
        os.makedirs(destination_path, exist_ok=True)

        print(f"Downloading model from lakeFS repo '{repo_name}' with commit_id:'{commit_id}' to '{destination_path}'...")
        repo = lakefs.repository(repo_name, client=self.lakefs_client).ref(commit_id)
        objects = repo.objects()
        for obj in objects:
            obj_path = os.path.join(destination_path, obj.path)
            local_file_dir = os.path.dirname(obj_path)
            if not os.path.exists(local_file_dir):
                os.makedirs(local_file_dir)
            try:
                # 使用 'wb' (write binary) 模式寫入，確保檔案內容正確無誤
                with open(obj_path, "wb") as f:
                    f.write(repo.object(obj.path).reader(mode='rb', pre_sign=False).read())
                success_msg = f"Downloaded: {obj.path} -> {obj_path}"
                print(success_msg)
            except Exception as e:
                print(f"下載失敗: {obj.path}, 錯誤: {e}")
        
        print(f"Model download completed. Local path: {destination_path}")
        return destination_path

    def download_from_lakefs_by_uri(self, lakefs_uri: str, destination_root: Optional[str] = None) -> str:
        """根據 lakeFS URI (lakefs://repo/commit_id[/optional_subdir]) 下載模型到本地。

        Args:
            lakefs_uri: 形如 lakefs://my-repo/abcdef123456/ (或包含子資料夾)
            destination_root: 下載根目錄，未提供則使用 self.download_from_lakefs_root
        Returns:
            本地下載後的完整路徑或錯誤訊息字串
        """
        try:
            if destination_root is None:
                destination_root = self.download_from_lakefs_root
            os.makedirs(destination_root, exist_ok=True)

            parsed = urlparse(lakefs_uri)
            if parsed.scheme != 'lakefs':
                return f"Error: invalid lakefs uri: {lakefs_uri}"
            repo_name = parsed.netloc
            path_parts = [p for p in parsed.path.strip('/').split('/') if p]
            if len(path_parts) == 0:
                return f"Error: URI missing commit id: {lakefs_uri}"
            commit_id = path_parts[0]
            subdir = '/'.join(path_parts[1:]) if len(path_parts) > 1 else ''

            local_base = os.path.join(destination_root, repo_name, commit_id)
            local_target = os.path.join(local_base, subdir) if subdir else local_base

            # 若已存在且非空則直接回傳
            if os.path.exists(local_target) and any(os.scandir(local_target)):
                print(f"Model already cached: {local_target}")
                return local_target

            os.makedirs(local_target, exist_ok=True)
            ref = lakefs.repository(repo_name, client=self.lakefs_client).ref(commit_id)
            prefix = subdir
            for obj in ref.objects(prefix=prefix):
                rel_path = obj.path[len(prefix):].lstrip('/') if prefix else obj.path
                target_file = os.path.join(local_target if subdir else local_base, rel_path)
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                try:
                    with open(target_file, 'wb') as f:
                        f.write(ref.object(obj.path).reader(mode='rb', pre_sign=False).read())
                except Exception as e:
                    print(f"Failed to download {obj.path}: {e}")
            return local_target
        except Exception as e:
            err = f"Error downloading by lakefs uri {lakefs_uri}: {e}"
            print(err)
            return err

    # def download_from_lakefs_by_uri(self, lakefs_uri: str, destination_root: Optional[str] = None) -> str:
    #     """
    #     根據 lakeFS URI 從一個特定的 commit 下載模型。
    #     Args:
    #         lakefs_uri (str): 格式為 "lakefs://repo/commit_id/folder"
    #         destination_root (str): 本地快取根目錄。
    #     Returns:
    #         str: 模型在本地的完整路徑。
    #     """
    #     if destination_root is None:
    #         destination_root = self.download_from_lakefs_root
        
    #     parsed = urlparse(lakefs_uri)
    #     repo_name = parsed.netloc
    #     path_parts = parsed.path.strip('/').split('/')
    #     commit_id = path_parts[0]
    #     folder_name = path_parts[1] if len(path_parts) > 1 else ""

    #     local_path = os.path.join(destination_root, repo_name, commit_id, folder_name)
    #     if os.path.exists(local_path) and os.listdir(local_path):
    #         print(f"模型已存在於本地快取: {local_path}")
    #         return local_path

    #     os.makedirs(local_path, exist_ok=True)
    #     print(f"正在從 LakeFS URI '{lakefs_uri}' 下載模型至 '{local_path}'...")
        
    #     try:
    #         ref = lakefs.repository(repo_name, client=self.lakefs_client).ref(commit_id)
    #         for obj in ref.objects(prefix=folder_name):
    #             relative_path = os.path.relpath(obj.path, folder_name)
    #             target_file_path = os.path.join(local_path, relative_path)
    #             os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                
    #             with open(target_file_path, "wb") as f:
    #                 f.write(ref.object(obj.path).reader(mode='rb').read())
            
    #         print("模型下載成功。")
    #         return local_path
    #     except Exception as e:
    #         download_error = f"Error downloading from lakeFS URI '{lakefs_uri}': {e}"
    #         print(download_error)
    #         return download_error

    def manage_inference_priority(
            self, 
            inference_task: str, 
            target_priority: Optional[int] = None,
            model_name_to_exclude: Optional[str] = None
        ) -> int:
        """
        智能管理並返回一個唯一的模型推理優先級。
        
        每次調用都會進行全面的優先級檢查和自動修正，確保：
        1. 所有模型的優先級都是唯一的
        2. 優先級序列是連續的（1, 2, 3, ...）
        3. 沒有重複或跳號的情況

        - 如果提供了 target_priority，則將該優先級插入，並將後續模型的優先級順延 (+1)。
        - 如果 target_priority 為 None，則自動分配為當前最大優先級 + 1。
        - 該操作會自動修正所有重複的優先級問題。

        Args:
            inference_task (str): 要管理的任務標識符。
            target_priority (Optional[int]): 期望的優先級。如果為 None，則自動分配。
            model_name_to_exclude (Optional[str]): 在操作中要排除的模型名稱和版本 (格式: "name:version")。
                                                    主要用於更新模型場景，防止自己跟自己比較。

        Returns:
            int: 分配或確認的唯一優先級。
                如果出錯，返回負數並打印錯誤信息。
        """
        try:
            if target_priority is not None and target_priority < 1:
                raise ValueError("優先級必須大於或等於 1。")
        except ValueError as ve:
            print(f"Invalid target_priority: {ve}")
            return -1
        print(f"正在為任務 '{inference_task}' 管理優先級...")
        
        filter_string = f"tags.stage = 'production' AND tags.inference_task = '{inference_task}'"
        
        try:
            versions = self.mlflow_client.search_model_versions(filter_string)
            
            # 獲取當前所有模型的 (優先級, 模型版本對象) 列表
            existing_models: List[Dict[str, Any]] = []
            models_without_priority: List[Dict[str, Any]] = []
            
            for v in versions:
                # 排除自身
                if model_name_to_exclude and f"{v.name}:{v.version}" == model_name_to_exclude:
                    continue
                
                p = v.tags.get("inference_priority")
                if p and p.isdigit():
                    existing_models.append({
                        "priority": int(p),
                        "version_obj": v
                    })
                else:
                    # 記錄沒有優先級標籤的模型，稍後給予優先級
                    models_without_priority.append({
                        "version_obj": v
                    })

            # 按優先級排序
            existing_models.sort(key=lambda x: x["priority"])
            
            # === 步驟 1: 檢查並修正重複的優先級 ===
            print("檢查優先級重複問題...")
            priority_counts = {}
            for model in existing_models:
                priority = model["priority"]
                if priority not in priority_counts:
                    priority_counts[priority] = []
                priority_counts[priority].append(model)
            
            # 找出重複的優先級並重新分配
            duplicates_found = False
            corrected_models = []
            next_available_priority = 1
            
            for priority in sorted(priority_counts.keys()):
                models_with_this_priority = priority_counts[priority]
                
                if len(models_with_this_priority) > 1:
                    duplicates_found = True
                    print(f"發現優先級 {priority} 有 {len(models_with_this_priority)} 個重複模型，進行修正...")
                    
                    # 重複優先級的模型需要重新分配
                    for i, model in enumerate(models_with_this_priority):
                        # 找到下一個可用的優先級
                        while any(m["priority"] == next_available_priority for m in corrected_models):
                            next_available_priority += 1
                        
                        # 如果是第一個，保持原優先級（如果可用）
                        if i == 0 and not any(m["priority"] == priority for m in corrected_models):
                            model["new_priority"] = priority
                            corrected_models.append({"priority": priority, "version_obj": model["version_obj"]})
                        else:
                            model["new_priority"] = next_available_priority
                            corrected_models.append({"priority": next_available_priority, "version_obj": model["version_obj"]})
                            next_available_priority += 1
                else:
                    # 沒有重複，保持原優先級
                    model = models_with_this_priority[0]
                    model["new_priority"] = priority
                    corrected_models.append({"priority": priority, "version_obj": model["version_obj"]})
            
            # === 步驟 2: 處理沒有優先級的模型 ===
            if models_without_priority:
                print(f"發現 {len(models_without_priority)} 個模型沒有優先級標籤，自動分配...")
                # 找到最大優先級
                max_priority = max([m["priority"] for m in corrected_models]) if corrected_models else 0
                
                for model in models_without_priority:
                    max_priority += 1
                    model["new_priority"] = max_priority
                    corrected_models.append({"priority": max_priority, "version_obj": model["version_obj"]})
                    print(f"  - 為模型 '{model['version_obj'].name}' (版本 {model['version_obj'].version}) 分配優先級: {max_priority}")
            
            # === 步驟 3: 更新所有需要修正的模型 ===
            all_models_to_update = []
            
            # 收集所有需要更新的模型（包括重複優先級修正和沒有優先級的）
            for model in existing_models + models_without_priority:
                if "new_priority" in model:
                    old_priority = model.get("priority", "無")
                    new_priority = model["new_priority"]
                    if old_priority != new_priority:
                        all_models_to_update.append({
                            "version_obj": model["version_obj"],
                            "old_priority": old_priority,
                            "new_priority": new_priority
                        })
            
            # 執行批量更新
            for update_info in all_models_to_update:
                v_obj = update_info["version_obj"]
                old_p = update_info["old_priority"]
                new_p = update_info["new_priority"]
                print(f"  - 更新模型 '{v_obj.name}' (版本 {v_obj.version}) 的優先級: {old_p} -> {new_p}")
                self.mlflow_client.set_model_version_tag(
                    name=v_obj.name,
                    version=v_obj.version,
                    key="inference_priority",
                    value=str(new_p)
                )
            
            # === 步驟 4: 重新整理模型列表，準備分配新優先級 ===
            # 重新獲取修正後的模型列表
            existing_models = corrected_models
            existing_models.sort(key=lambda x: x["priority"])
            
            final_priority: int
            models_to_update: List[Dict[str, Any]] = []

            if target_priority is None:
                # --- 場景 A: 自動分配到末位 ---
                if not existing_models:
                    final_priority = 1
                else:
                    final_priority = existing_models[-1]["priority"] + 1
                        
                print(f"未指定優先級，自動分配為: {final_priority}")

            else:
                # --- 場景 B: 插入指定優先級 ---
                final_priority = target_priority
                is_priority_exist = any(m["priority"] == final_priority for m in existing_models)

                if not is_priority_exist:
                    print(f"優先級 '{final_priority}' 未被佔用，無需重新排序。")
                else:
                    print(f"優先級 '{final_priority}' 已存在，觸發重新排序...")
                    # 找到所有需要順延的模型
                    for model in existing_models:
                        if model["priority"] >= final_priority:
                            models_to_update.append(model)
                    
                    # 為了避免衝突，我們從後往前更新
                    models_to_update.sort(key=lambda x: x["priority"], reverse=True)

                    for model in models_to_update:
                        old_p = model["priority"]
                        new_p = old_p + 1
                        v_obj = model["version_obj"]
                        print(f"  - 更新模型 '{v_obj.name}' (版本 {v_obj.version}) 的優先級: {old_p} -> {new_p}")
                        self.mlflow_client.set_model_version_tag(
                            name=v_obj.name,
                            version=v_obj.version,
                            key="inference_priority",
                            value=str(new_p)
                        )
            
            if duplicates_found or models_without_priority or all_models_to_update:
                print(f"優先級管理完成，所有模型的優先級已確保唯一且連續。")
            
            return final_priority

        except Exception as e:
            error_msg = f"Error managing inference priority: {e}"
            print(error_msg)
            return -1

    def register_lakefs_to_mlflow(self, 
                                  registered_name: str, 
                                  lakefs_repo: str, 
                                  task: str, 
                                  engine: str, 
                                  model_params: float,
                                  set_priority_to_one: Optional[bool] = False,                                                        
                                  commit_id: Optional[str] = None, 
                                  lakefs_branch: Optional[str] = "main",
                                  version_description: Optional[str] = "", 
                                  stage: Optional[str] = None,) -> dict:
        """
        將 lakeFS 中已提交的模型版本註冊到 MLflow 模型註冊表中。

        Args:
            registered_name (str): 在 MLflow 中註冊的模型名稱。
            lakefs_repo (str): lakeFS 儲存庫名稱。
            task (str): 模型任務類型，需嚴格符合 inference.yaml 中的 task_to_models 定義。
            engine (str): 模型引擎類型，如 "transformers" 或 "vllm" 或 "custom"。
            model_params (float): 模型參數量，用於估算 VRAM。(單位為B(10億），如 7B模型則填 7)
            set_priority_to_one (bool): 是否將該模型的推理優先級設為最高(1)，默認為 False 時則設為目前所有同任務模型中的最後
            commit_id (str): lakeFS 中的 commit ID，指向一個不可變的模型版本。默認為 None，表示使用指定分支的最新 commit。
            lakefs_branch (str): lakeFS 分支名稱，默認為 "main"。
            version_description (str): 版本描述信息。
            stage (str):   
                目標階段，有以下四種：
                1. "none": 默認階段
                2. "production": 正式生產階段
                3. "staging": 預備階段
                4. "archived": 已封存階段
                若指定，則在註冊後自動將模型版本轉移到該階段。 
        Returns:
            Dict: 註冊的模型版本資訊或錯誤信息
        """
        # 物理路徑指向一個不可變的 commit
        if commit_id is None:
            # 獲取分支的最新 commit ID
            try:
                commit_id = lakefs.repository(lakefs_repo, client=self.lakefs_client).branch(lakefs_branch).get_commit().id
                print(f"Using latest commit ID from branch '{lakefs_branch}': {commit_id}")
            except Exception as e:
                error_msg = f"Failed to fetch latest commit ID from branch '{lakefs_branch}': {e}"
                print(error_msg)
                return {"error": error_msg}
        model_uri = f"lakefs://{lakefs_repo}/{commit_id}/"
        stage = stage.lower() if stage else "none"

        mlflow.set_experiment(experiment_name=registered_name)
        with mlflow.start_run(run_name=f"Register-{registered_name}", 
                              description=version_description) as run:
            run_id = run.info.run_id
            
            # 建立一個虛擬 artifact 來滿足 MLflow 的註冊要求
            dummy_model_path = os.path.join(self.models_tmp_root, f"dummy_for_{run_id}")
            os.makedirs(dummy_model_path,exist_ok=True)
            readme_content = (
                f"This MLflow model version is a pointer to a model stored in lakeFS.\n"
                f"Physical Location: {model_uri}\n"
                f"Repository: {lakefs_repo}\n"
                f"Commit ID: {commit_id}\n"
            )
            # (dummy_model_path / "model_pointer.md").write_text(readme_content)
            with open(os.path.join(dummy_model_path, "model_pointer.md"), "w") as f:
                f.write(readme_content)
       
            try:
                mlflow.log_artifact(str(dummy_model_path), artifact_path="model_pointer")
            except Exception as e:
                log_error = f"Error logging dummy artifact to MLflow: {e}"
                print(log_error)                
                return {"error": log_error}
            shutil.rmtree(dummy_model_path) # 清理臨時檔案
            artifact_source_uri = f"mlflow-artifacts:/0/{run_id}/artifacts/model_pointer"

            tags_to_register = {
                "physical_path": model_uri,
                "model_type": task, # model_type 仍然保留，用於一般分類
                "lakefs_repo": lakefs_repo,
                "lakefs_commit_id": commit_id,
                "stage": stage
            }
            # 合併推理相關的標籤
            model_params = int(model_params * 1e9) # 將B(10億）轉為實際數字    
            
            # 使用 estimate_and_prepare_tags 獲取完整的估算標籤
            model_info = {
                "params": model_params,
                "model_type": task,
                "precision": "fp16"  # 預設精度，可以根據需要調整
            }
            estimation_tags = self.estimator.estimate_and_prepare_tags(model_info)
            
            # 從估算標籤中獲取 VRAM 值（向後兼容）
            estimated_vram_gb = float(estimation_tags.get('est_estimated_total_vram_gb', '4.0'))
            
            if set_priority_to_one:
                priority = 1
                checked_priority = self.manage_inference_priority(inference_task=task, 
                                               target_priority=priority)
                if checked_priority != priority:
                    error_msg = f"Failed to set priority to 1 for task '{task}'."
                    print(error_msg)
                    return {"error": error_msg}
            else:
                priority = self.manage_inference_priority(inference_task=task, 
                                               target_priority=None)
                if priority < 1:
                    error_msg = f"Failed to assign priority for task '{task}'."
                    print(error_msg)
                    return {"error": error_msg}
            if task in self.inference_task_rules and engine in self.inference_engines:
                    tags_to_register["inference_task"] = task
                    tags_to_register["inference_engine"] = engine
                    tags_to_register["inference_estimated_vram_gb"] = str(estimated_vram_gb)
                    tags_to_register["inference_priority"] = str(priority)
                    
                    # 合併完整的 VRAM 估算標籤
                    tags_to_register.update(estimation_tags)
            else:
                error_msg = f"Task '{task}' or engine '{engine}' is not recognized based on the inference configuration."
                print(error_msg)
                return {"error": error_msg}

            try:
                mv = mlflow.register_model(
                    model_uri=artifact_source_uri,
                    name=registered_name,
                    tags=tags_to_register, 
                )
                self.mlflow_client.update_model_version(
                    name=registered_name,
                    version=mv.version,
                    description=version_description
                )
            except Exception as e:
                register_error = f"Error registering model to MLflow: {e}"
                print(register_error)
                return {"error": register_error}
            
            print(f"Model '{registered_name}' version {mv.version} registered successfully.")
            print(f"  - MLflow Run ID: {run_id}")
            print(f"  - LakeFS Commit ID: {commit_id}")
            if stage in ["production", "staging", "archived", "none"]:
                # 轉移模型版本到指定階段
                if stage == "production":
                    archive_existing_versions = True
                else:
                    archive_existing_versions = False
                print(f"Transitioning model '{registered_name}' version {mv.version} to stage '{stage}'...")
                self.model_transition_stage_on_mlflow(model_name=registered_name, 
                                                    version=mv.version, 
                                                    stage=stage, 
                                                    archive_existing_versions=archive_existing_versions)
            
            return {
                "name": mv.name,
                "version": mv.version,
                "physical_path": mv.tags.get("physical_path"),
                "lakefs_commit_id": mv.tags.get("lakefs_commit_id"),
                "run_id": run_id, 
                "stage": stage, 
                "inference_task": mv.tags.get("inference_task"),
                "inference_engine": mv.tags.get("inference_engine"),
                "inference_estimated_vram_gb": mv.tags.get("inference_estimated_vram_gb"),
                "inference_priority": mv.tags.get("inference_priority"),
            }
    def register_ollama_to_mlflow(self, 
                                  local_model_name: str,
                                  task: str,
                                  model_params: Optional[float] = None, 
                                  set_priority_to_one: Optional[bool] = False, 
                                  registered_name: Optional[str] = None,
                                  version_description: Optional[str] = "", 
                                  stage: Optional[str] = None) -> dict:
        """
        將 Ollama 模型註冊到 MLflow 模型註冊表中。
        Args:
            local_model_name (str): 本地 Ollama 模型名稱。
            task (str): 模型任務類型，需嚴格符合 inference.yaml 中的 task_to_models 定義。
            model_params (float): 模型參數量，用於估算 VRAM。(單位為B(10億），如 7B模型則填7)不填則使用 ollama api 回傳的參數量
            set_priority_to_one (bool): 是否將該模型的推理優先級設為最高(1)，默認為 False 時則設為目前所有同任務模型中的最後
            registered_name (str): 在 MLflow 中註冊的模型名稱，若為 None 則使用 local_model_name。
            version_description (str): 版本描述信息。
            stage (str): 
                目標階段，有以下四種：
                1. "none": 默認階段
                2. "production": 正式生產階段
                3. "staging": 預備階段
                4. "archived": 已封存階段
                若指定，則在註冊後自動將模型版本轉移到該階段。
            
        Returns:
            Dict: 註冊的模型版本資訊 或錯誤訊息
        """
        # 檢查本地是否有該 Ollama 模型
        all_model_info = self.list_local_models(model_source="ollama")
        target_model_info: dict = {}
        for m in all_model_info:
            if m["model_name"] == local_model_name:
                print(f"Found Ollama model: {m}")
                target_model_info = m
                break
        if not target_model_info:
            not_found_msg = f"Ollama model '{local_model_name}' not found locally. Please pull it first."
            print(not_found_msg)
            return {"error": not_found_msg}
        
        if registered_name is None:
            registered_name = local_model_name
        stage = stage.lower() if stage else "none"

        if not model_params:
            param_size_str = target_model_info.get("parameter_size", "0")
            print(f"Parsing parameter size: '{param_size_str}'")
            
            # 更安全的參數量解析（不使用正則表達式）
            try:
                # 將字串轉為大寫並去除空白
                param_str_upper = param_size_str.strip().upper()
                
                # 分離數字和單位
                number_str = ""
                unit = ""
                
                for i, char in enumerate(param_str_upper):
                    if char.isdigit() or char == '.':
                        number_str += char
                    elif char in ['K', 'M', 'B']:
                        # 找到單位，取剩餘部分作為單位（允許 "7B" 或 "7.5MB" 等格式）
                        unit = char
                        break
                    elif char == ' ':
                        # 跳過空白
                        continue
                    else:
                        # 遇到其他字符，嘗試繼續解析
                        continue
                
                # 如果沒有找到數字，設為預設值
                if not number_str:
                    print(f"Warning: Could not find numeric value in '{param_size_str}', using default 1.0B")
                    model_params = 1.0
                else:
                    number = float(number_str)
                    
                    # 如果沒有單位，預設為 B
                    if not unit:
                        unit = 'B'
                    
                    # 根據單位轉換
                    if unit == 'K':
                        model_params = number / 1e6  # K 轉為 B
                    elif unit == 'M':
                        model_params = number / 1000  # M 轉為 B
                    else:  # unit == 'B'
                        model_params = number
                    
                    print(f"Successfully parsed '{param_size_str}' as {model_params:.3f}B")
                    
            except (ValueError, AttributeError) as e:
                print(f"Error parsing parameter size '{param_size_str}': {e}, using default 1.0B")
                model_params = 1.0
        # 根據 Ollama 量化級別推斷精度
        quantization_level = target_model_info.get("quantization_level", "")
        precision = self._infer_precision_from_quantization(quantization_level)
        print(f"Inferred precision '{precision}' from quantization level '{quantization_level}'")
        
        model_params_actual = int(model_params * 1e9) # 將B(10億）轉為實際數字    
        
        # 使用 estimate_and_prepare_tags 直接獲取 MLflow 標籤格式
        ollama_estimation_tags = self.estimator.estimate_and_prepare_tags({
            "params": model_params_actual,
            "model_type": "text-generation-ollama", 
            "precision": precision,
            "quantization_level": quantization_level
        })
        
        # 從估算標籤中獲取 VRAM 值（向後相容）
        estimated_vram_gb = float(ollama_estimation_tags.get('est_estimated_total_vram_gb', '4.0'))

        mlflow.set_experiment(experiment_name=registered_name)
        with mlflow.start_run(run_name=f"Register-{registered_name}", 
                              description=version_description) as run:
            run_id = run.info.run_id
            
            # 建立一個虛擬 artifact 來滿足 MLflow 的註冊要求
            dummy_model_path = os.path.join(self.models_tmp_root, f"dummy_for_{run_id}")
            os.makedirs(dummy_model_path,exist_ok=True)
            readme_content = (
                f"This MLflow model version is a pointer to an Ollama model.\n"
                f"Model Name: {local_model_name}\n"
            )
            # (dummy_model_path / "model_pointer.md").write_text(readme_content)
            with open(os.path.join(dummy_model_path, "model_pointer.md"), "w") as f:
                f.write(readme_content)
            
        
            try:
                mlflow.log_artifact(str(dummy_model_path), artifact_path="model_pointer")
            except Exception as e:
                print(f"Error logging dummy artifact to MLflow: {e}")
                
                return e
            shutil.rmtree(dummy_model_path) # 清理臨時檔案
            artifact_source_uri = f"mlflow-artifacts:/0/{run_id}/artifacts/model_pointer"

            tags_to_register = {
                        "model_type": task,
                        "ollama_model_name": local_model_name, 
                        "ollama_model_size_gb": str(target_model_info.get("model_size_gb", "")) + "GB", 
                        "ollama_parameter_size": target_model_info.get("parameter_size", ""), 
                        "ollama_quantization_level": target_model_info.get("quantization_level", ""),
                        "ollama_inferred_precision": precision,
                        "ollama_parsed_params_b": str(model_params),
                        "stage": stage, 
            }
            # 合併推理相關的標籤
            if set_priority_to_one:
                priority = 1
                checked_priority = self.manage_inference_priority(inference_task=task, 
                                               target_priority=priority)
                if checked_priority != priority:
                    error_msg = f"Failed to set priority to 1 for task '{task}'."
                    print(error_msg)
                    return {"error": error_msg}
            else:
                priority = self.manage_inference_priority(inference_task=task, 
                                               target_priority=None)
                if priority < 1:
                    error_msg = f"Failed to assign priority for task '{task}'."
                    print(error_msg)
                    return {"error": error_msg}
            if task in self.inference_task_rules:
                    tags_to_register["inference_task"] = task
                    tags_to_register["inference_engine"] = "ollama"
                    tags_to_register["inference_estimated_vram_gb"] = str(estimated_vram_gb)
                    tags_to_register["inference_priority"] = str(priority)
                    
                    # 合併完整的 VRAM 估算標籤
                    tags_to_register.update(ollama_estimation_tags)
            else:
                error_msg = f"Task '{task}' is not recognized based on the inference configuration."
                print(error_msg)
                return {"error": error_msg}
            
            
            try:                
                mv = mlflow.register_model(
                    model_uri=artifact_source_uri,
                    name=registered_name,
                    tags=tags_to_register,
                )
                self.mlflow_client.update_model_version(
                    name=registered_name,
                    version=mv.version,
                    description=version_description
                )
            except Exception as e:
                print(f"Error registering model to MLflow: {e}")
                return e
            
            print(f"Ollama Model '{registered_name}' version {mv.version} registered successfully.")
            print(f"  - MLflow Run ID: {run_id}")
            if stage in ["production", "staging", "archived", "none"]:
                # 轉移模型版本到指定階段
                if stage == "production":
                    archive_existing_versions = True
                else:
                    archive_existing_versions = False
                print(f"Transitioning model '{registered_name}' version {mv.version} to stage '{stage}'...")
                self.model_transition_stage_on_mlflow(model_name=registered_name, 
                                                    version=mv.version, 
                                                    stage=stage, 
                                                    archive_existing_versions=archive_existing_versions)
            return {
                "name": mv.name,
                "version": mv.version,
                "ollama_model_name": mv.tags.get("ollama_model_name"),
                "run_id": run_id
            }

    def list_mlflow_models(self, show_all:bool=False) -> list:
        """
        從 MLflow 獲取已註冊的模型及其版本信息。
        Args:
            show_all (bool): 是否顯示所有版本，預設為 False，只顯示最新版本。
        Returns:
            list: 包含模型詳細資訊的列表。  
            格式如下: 
            [
                {
                    "name": "model_name",
                    "versions": [
                        {
                            "version": "1",
                            "stage": "Production",
                            "description": "First production version",
                            "run_id": "abcd1234",
                            "creation_timestamp": "2023-10-01 12:00:00",
                            "last_updated_timestamp": "2023-10-02 12:00:00",
                            "tags": {"key": "value"}
                        },
                        ...
                    ]
                },
                ...
            ]
        """
        models = []
        for rm in self.mlflow_client.search_registered_models():
            model_info = {"name": rm.name, "alises": rm.aliases, "versions": []}
            versions_to_process = rm.latest_versions if not show_all else self.mlflow_client.search_model_versions(f"name='{rm.name}'")
            
            for mv in versions_to_process:
                model_info["versions"].append({
                    "version": mv.version,
                    "stage": mv.tags.get("stage", "None"),
                    "description": mv.description,
                    "creation_timestamp": datetime.datetime.fromtimestamp(mv.creation_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "last_updated_timestamp": datetime.datetime.fromtimestamp(mv.last_updated_timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "tags": mv.tags
                })
            models.append(model_info)
        return models

    def get_model_details_from_mlflow(self, model_name: str="", all_version: bool=False) -> dict:
        """
        根據模型名和階段獲取模型詳細信息（尤其是物理路徑）。
        Args:
            model_name (str): 在 MLflow 中註冊的模型名稱。
            all_version (bool): 是否返回所有版本的資訊，預設為 False，只返回最新版本。
        Returns:
            dict: 包含模型詳細資訊的字典，如果找不到則返回錯誤訊息。
        """
        all_models = self.list_mlflow_models(show_all=True)
        try:
            for model in all_models:
                if model["name"] == model_name:
                    if all_version:
                        return model
                    else:
                        # 返回最新版本
                        latest_version = max(model["versions"], key=lambda x: int(x["version"]))
                        return latest_version
            print(f"Model '{model_name}' not found in MLflow Registry.")
            return {"error": f"Model '{model_name}' not found in MLflow Registry."}
            
                        
        except Exception as e:
            print(f"Error fetching model details for '{model_name}': {e}")
            return {"error": str(e)}
        
    def model_transition_stage_on_mlflow(self, model_name: str, version: str, stage: str, archive_existing_versions: bool = False) -> str:
        """
        將指定模型版本轉移到指定階段。
        Args:
            model_name (str): 在 MLflow 中註冊的模型名稱。
            version (str): 模型版本號。
            stage (str): 目標階段，有以下四種：
                1. "none": 默認階段
                2. "production": 正式生產階段
                3. "staging": 預備階段
                4. "archived": 已封存階段
            archive_existing_versions (bool): 
                - 當 stage 為 "production" 時，此項必須為 True 才能自動封存當前的 production 模型。
                - 當 stage 為 "staging" 時，若為 True，則會封存其他已在 staging 的模型。
                預設為 False。
        Returns:
            str: 成功或錯誤信息。
        """
        # 1. 輸入驗證 (Input Validation)
        try:
            valid_stages = {"production", "staging", "archived", "none"}
            stage_lower = stage.lower()
            if stage_lower not in valid_stages:
                raise ValueError(f"無效的階段 '{stage}'。請使用 {valid_stages} 其中之一。")
        except ValueError as ve:
            return f"Invalid stage: {ve}"

        # 2. 預先收集模型版本資訊
        try:
            all_models = self.list_mlflow_models(show_all=True)
            target_model_info = next((m for m in all_models if m["name"] == model_name), None)

            if not target_model_info:
                raise ValueError(f"找不到模型 '{model_name}'。")
        except ValueError as ve:
            return f"Error: {ve}"
        model_versions = target_model_info.get("versions", [])
        target_mv = None # 目標版本
        current_prod_mv = [] # 當前所有在 Production 的版本
        other_versions_in_target_stage = [] # 其他在目標階段的版本

        for mv in model_versions:
            current_stage = mv.get("tags", {}).get("stage", "none").lower()
            if mv["version"] == version: # 找到目標版本
                target_mv = mv
                target_mv["current_stage"] = current_stage
            
            if current_stage == "production": # 收集當前所有在 Production 的版本
                current_prod_mv.append(mv)
            
            if current_stage == stage_lower: # 收集所有在目標階段的版本
                other_versions_in_target_stage.append(mv)

        if not target_mv:
            try:
                raise ValueError(f"在模型 '{model_name}' 中找不到版本 '{version}'。")
            except ValueError as ve:
                return f"Error: {ve}"
        
        # 3. 執行業務邏輯與安全檢查
        # 檢查點 3.1: 如果已經在目標階段，無需操作 (Idempotency)
        # if target_mv["current_stage"] == stage_lower:
        #     return f"模型 '{model_name}' 版本 {version} 已經在階段 '{stage_lower}'，無需轉移。"

        # 檢查點 3.2: 禁止將 Production 模型直接降級或封存
        if target_mv["current_stage"] == "production" and stage_lower in ["staging", "archived", "none"]:
            try:
                raise ValueError(
                    f"禁止將 Production 模型 '{model_name}' 版本 {version} 直接降級或封存。"
                    f"請直接將另一個版本設定為 Production。"
                )
            except ValueError as ve:
                return f"Error: {ve}"
        # 4. 執行狀態轉換
        try:
            # --- 核心邏輯：處理 Production ---
            if stage_lower == "production":
                # 如果已有其他版本是 Production(可能因不明原因出現多個tag為production的版本)
                if current_prod_mv: # 有其他版本在 Production 則封存
                    if not archive_existing_versions:
                        archive_existing_versions = True
                        print(f"警告: 模型 '{model_name}' 已有其他版本在 Production。"
                              f"根據 'archive_existing_versions' 規則，將自動封存其他 Production 版本。")
                    for mv_to_archive in current_prod_mv:
                        if mv_to_archive["version"] != version: # 確保不會動到目標版本自己
                            print(f"根據 'archive_existing_versions' 規則，將版本 {mv_to_archive['version']} 從 Production 封存...")
                            self.mlflow_client.set_model_version_tag(
                                name=model_name,
                                version=mv_to_archive["version"],
                                key="stage",
                                value="archived"
                            )
                
                # 將目標版本設為 Production
                print(f"將版本 {version} 設定為 Production...")
                self.mlflow_client.set_model_version_tag(name=model_name, version=version, key="stage", value="production")
                
                # 因為alias是唯一的，只在要設為Production時設置
                self.mlflow_client.set_registered_model_alias(name=model_name, 
                                                              alias="production", 
                                                              version=version)
                return f"模型 '{model_name}' 版本 {version} 成功轉移到 Production。舊版本（如有）已被封存。"

            # --- 核心邏輯：處理 Staging, Archived, None ---
            else:
                # 如果設定要封存同一階段的其他版本
                if archive_existing_versions:
                    for mv_to_archive in other_versions_in_target_stage:
                        # 確保不會動到目標版本自己
                        if mv_to_archive["version"] != version and stage_lower != "archived":
                            print(f"根據 'archive_existing_versions' 規則，將版本 {mv_to_archive['version']} 從 {stage_lower} 封存...")
                            self.mlflow_client.set_model_version_tag(
                                name=model_name,
                                version=mv_to_archive["version"],
                                key="stage",
                                value="archived"
                            )

                # 轉換目標版本
                print(f"將版本 {version} 轉移到 {stage_lower}...")
                self.mlflow_client.set_model_version_tag(name=model_name, version=version, key="stage", value=stage_lower)
                return f"模型 '{model_name}' 版本 {version} 成功轉移到階段 '{stage_lower}'。"

        except Exception as e:
            print(f"在轉移模型 '{model_name}' 版本 {version} 到 '{stage}' 時發生未知錯誤: {e}")
            return f"在轉移模型 '{model_name}' 版本 {version} 到 '{stage}' 時發生未知錯誤: {e}"

model_manager = ModelManager()