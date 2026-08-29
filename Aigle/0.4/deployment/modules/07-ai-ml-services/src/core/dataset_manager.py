# src/core/dataset_manager.py
import os
import shutil
import lakefs
from lakefs.client import Client
from lakefs import exceptions
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path
import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
from datasets import load_dataset
from .config import config

class DatasetManager:
    """
    負責數據集的上傳、下載、註冊和版本管理。
    支援 lakeFS 和 MLflow 整合，提供完整的數據集生命週期管理。
    """
    def __init__(self):
        # lakeFS 設定
        self.lakefs_client = Client(host=config.get_config("lakefs", "endpoint"),
                                    username=config.get_config("lakefs", "access_key"), 
                                    password=config.get_config("lakefs", "secret_key"))
        self.lakefs_datasets_storage_namespace = config.get_config("lakefs", "dataset_storage_namespace")
        
        # MLflow 設定
        self.mlflow_uri = config.get_config("mlflow", "tracking_uri")
        mlflow.set_tracking_uri(self.mlflow_uri)
        self.mlflow_client = MlflowClient(tracking_uri=self.mlflow_uri)
        
        # 本地存儲路徑設定
        self.datasets_tmp_root = os.path.abspath(config.get_config("storage", "datasets_tmp_root"))
        os.makedirs(self.datasets_tmp_root, exist_ok=True)
        
        # 從 lakeFS 下載的數據集存放路徑
        self.download_from_lakefs_root = os.path.abspath(
            config.get_config("storage", "datasets_from_lakefs_root")
        )
        os.makedirs(self.download_from_lakefs_root, exist_ok=True)
        
        # HuggingFace 設定
        self.hf_token = config.get_config("huggingface", "hf_token")

    def _detect_dataset_type(self, dataset_sample) -> str:
        """
        檢測數據集類型（文本、圖像、音頻、視頻等）
        
        Args:
            dataset_sample: 數據集的一個樣本
            
        Returns:
            str: 數據集類型
        """
        if not dataset_sample:
            return "unknown"
            
        # 檢查是否有圖像字段
        image_fields = ['image', 'img', 'picture', 'photo', 'pixel_values']
        audio_fields = ['audio', 'sound', 'speech', 'waveform', 'audio_file']
        video_fields = ['video', 'clip', 'frames', 'video_file']
        
        sample_keys = set(dataset_sample.keys()) if isinstance(dataset_sample, dict) else set()
        
        if any(field in sample_keys for field in image_fields):
            return "image"
        elif any(field in sample_keys for field in audio_fields):
            return "audio"  
        elif any(field in sample_keys for field in video_fields):
            return "video"
        elif any(key in ['text', 'sentence', 'content', 'input_text', 'target_text'] for key in sample_keys):
            return "text"
        else:
            return "multimodal"

    def _save_multimedia_files(self, dataset_split, split_name: str, destination_path: str, max_samples: int = 1000) -> dict:
        """
        保存多媒體文件（圖像、音頻、視頻等）到本地
        
        Args:
            dataset_split: 數據集分割
            split_name: 分割名稱
            destination_path: 目標路徑
            max_samples: 最大處理樣本數量
            
        Returns:
            dict: 多媒體文件統計信息
        """
        multimedia_stats = {}
        multimedia_dir = os.path.join(destination_path, "multimedia", split_name)
        os.makedirs(multimedia_dir, exist_ok=True)
        
        image_count = audio_count = video_count = 0
        
        try:
            for idx, sample in enumerate(dataset_split):
                if isinstance(sample, dict):
                    for key, value in sample.items():
                        # 處理圖像
                        if key in ['image', 'img', 'picture', 'photo'] and value is not None:
                            try:
                                # 如果是 PIL Image 對象
                                if hasattr(value, 'save'):
                                    image_path = os.path.join(multimedia_dir, f"image_{idx}_{key}.png")
                                    value.save(image_path)
                                    image_count += 1
                                # 如果是文件路徑或 bytes
                                elif isinstance(value, (str, bytes)):
                                    image_path = os.path.join(multimedia_dir, f"image_{idx}_{key}.png")
                                    if isinstance(value, str) and os.path.exists(value):
                                        shutil.copy2(value, image_path)
                                        image_count += 1
                                    elif isinstance(value, bytes):
                                        with open(image_path, 'wb') as f:
                                            f.write(value)
                                        image_count += 1
                            except Exception as e:
                                print(f"Error saving image {idx}: {e}")
                        
                        # 處理音頻
                        elif key in ['audio', 'sound', 'speech'] and value is not None:
                            try:
                                audio_path = os.path.join(multimedia_dir, f"audio_{idx}_{key}")
                                if isinstance(value, dict) and 'array' in value:
                                    # HuggingFace 音頻格式
                                    import numpy as np
                                    audio_array = np.array(value['array'])
                                    sample_rate = value.get('sampling_rate', 16000)
                                    
                                    # 保存為 wav 文件（需要 soundfile 或 scipy）
                                    try:
                                        import soundfile as sf
                                        sf.write(f"{audio_path}.wav", audio_array, sample_rate)
                                        audio_count += 1
                                    except ImportError:
                                        # 如果沒有 soundfile，保存為 numpy 文件
                                        np.save(f"{audio_path}.npy", audio_array)
                                        audio_count += 1
                                elif isinstance(value, (str, bytes)):
                                    if isinstance(value, str) and os.path.exists(value):
                                        shutil.copy2(value, f"{audio_path}.wav")
                                        audio_count += 1
                                    elif isinstance(value, bytes):
                                        with open(f"{audio_path}.wav", 'wb') as f:
                                            f.write(value)
                                        audio_count += 1
                            except Exception as e:
                                print(f"Error saving audio {idx}: {e}")
                        
                        # 處理視頻  
                        elif key in ['video', 'clip', 'frames'] and value is not None:
                            try:
                                video_path = os.path.join(multimedia_dir, f"video_{idx}_{key}")
                                if isinstance(value, (str, bytes)):
                                    if isinstance(value, str) and os.path.exists(value):
                                        shutil.copy2(value, f"{video_path}.mp4")
                                        video_count += 1
                                    elif isinstance(value, bytes):
                                        with open(f"{video_path}.mp4", 'wb') as f:
                                            f.write(value)
                                        video_count += 1
                            except Exception as e:
                                print(f"Error saving video {idx}: {e}")
                                
                # 限制處理的樣本數量以避免過大的數據集
                if idx >= max_samples:  
                    print(f"Processed first {max_samples} samples for multimedia extraction.")
                    break
                    
        except Exception as e:
            print(f"Error processing multimedia files: {e}")
        
        multimedia_stats = {
            'images': image_count,
            'audios': audio_count, 
            'videos': video_count,
            'multimedia_dir': multimedia_dir
        }
        
        return multimedia_stats

    def _process_and_save_dataset(self, dataset, destination_path: str, dataset_name: str, 
                                 split: Optional[str] = None, extract_multimedia: bool = True, 
                                 max_samples: int = 1000) -> dict:
        """
        處理並保存數據集，支援多種格式和多媒體內容
        
        Args:
            dataset: 加載的數據集
            destination_path: 目標路徑
            dataset_name: 數據集名稱
            split: 指定的分割
            extract_multimedia: 是否提取多媒體文件
            max_samples: 提取多媒體文件的最大樣本數量
            
        Returns:
            dict: 處理結果統計信息
        """
        processing_info = {
            'dataset_type': 'unknown',
            'total_samples': 0,
            'multimedia_count': 0,
            'output_formats': [],
            'multimedia_stats': {}
        }
        
        try:
            # 如果指定了分割，只處理該分割
            if split:
                dataset_splits = {split: dataset}
            else:
                # 處理所有分割
                if hasattr(dataset, 'keys'):
                    dataset_splits = {k: v for k, v in dataset.items()}
                else:
                    dataset_splits = {'train': dataset}
            
            total_samples = 0
            
            for split_name, split_data in dataset_splits.items():
                try:
                    # 檢測數據集類型
                    if len(split_data) > 0:
                        sample = split_data[0]
                        dataset_type = self._detect_dataset_type(sample)
                        processing_info['dataset_type'] = dataset_type
                    
                    split_samples = len(split_data)
                    total_samples += split_samples
                    
                    print(f"Processing {split_name} split with {split_samples} samples...")
                    
                    # 始終嘗試保存為 JSON（包含元數據）
                    try:
                        json_path = os.path.join(destination_path, f"{split_name}.json")
                        split_data.to_json(json_path)
                        processing_info['output_formats'].append('json')
                        print(f"Saved {split_name} as JSON")
                    except Exception as e:
                        print(f"Warning: Could not save {split_name} as JSON: {e}")
                    
                    # 嘗試保存為 CSV（如果支持且是結構化數據）
                    try:
                        if hasattr(split_data, 'to_csv'):
                            csv_path = os.path.join(destination_path, f"{split_name}.csv") 
                            split_data.to_csv(csv_path)
                            if 'csv' not in processing_info['output_formats']:
                                processing_info['output_formats'].append('csv')
                            print(f"Saved {split_name} as CSV")
                    except Exception as e:
                        print(f"Info: Could not save {split_name} as CSV (normal for multimedia datasets): {e}")
                    
                    # 保存為 Parquet 格式（更適合大數據集）
                    try:
                        parquet_path = os.path.join(destination_path, f"{split_name}.parquet")
                        split_data.to_parquet(parquet_path)
                        if 'parquet' not in processing_info['output_formats']:
                            processing_info['output_formats'].append('parquet')
                        print(f"Saved {split_name} as Parquet")
                    except Exception as e:
                        print(f"Info: Could not save {split_name} as Parquet: {e}")
                    
                    # 處理多媒體內容
                    if extract_multimedia and dataset_type in ['image', 'audio', 'video', 'multimodal']:
                        print(f"Extracting multimedia content from {split_name}...")
                        multimedia_stats = self._save_multimedia_files(split_data, split_name, destination_path, max_samples)
                        processing_info['multimedia_stats'][split_name] = multimedia_stats
                        processing_info['multimedia_count'] += sum(multimedia_stats.get(k, 0) for k in ['images', 'audios', 'videos'])
                        
                except Exception as e:
                    print(f"Error processing split {split_name}: {e}")
                    continue
            
            processing_info['total_samples'] = total_samples
            
            # 如果沒有成功的輸出格式，至少保存原始緩存
            if not processing_info['output_formats']:
                processing_info['output_formats'] = ['cache']
                
        except Exception as e:
            print(f"Error in dataset processing: {e}")
            
        return processing_info

    def _ensure_lakefs_repo(self, repo_name: str, branch_name: Optional[str] = "main"):
        """
        使用 lakeFS API 確保儲存庫以及分支存在，若不存在則創建，已存在則返回現有的儲存庫資訊。
        
        Args:
            repo_name (str): 儲存庫名稱
            branch_name (Optional[str]): 預設分支名稱，默認為 "main"
            
        Returns:
            Repository or branch information, or None if failed
        """
        storage_namespace = f"{self.lakefs_datasets_storage_namespace}/{repo_name}"
        
        try:
            # 嘗試創建新的儲存庫
            repo = lakefs.repository(repo_name, client=self.lakefs_client).create(
                storage_namespace, default_branch="main"
            )
            print(f"Successfully created lakeFS repository: {repo}")
            return repo
            
        except exceptions.ConflictException as e:
            print(f"Repository {repo_name} already exists: {e}")
            
            # 儲存庫已存在，檢查並創建分支（如果需要）
            for repo in lakefs.repositories(client=self.lakefs_client):
                if repo.id == repo_name:
                    print(f"Repo info: {repo}")
                    break
                    
            try:
                branch = lakefs.repository(repo_name, client=self.lakefs_client).branch(branch_name).create(
                    source_reference="main"
                )
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

    def download_dataset(self, 
                        dataset_source: str,
                        dataset_name: str,
                        destination_path: Optional[str] = None,
                        split: Optional[str] = None,
                        config_name: Optional[str] = None,
                        extract_multimedia: Optional[bool] = True,
                        max_samples: Optional[int] = 1000) -> str:
        """
        從 HuggingFace Hub 或其他來源下載數據集到指定路徑，支援多媒體數據集
        
        Args:
            dataset_source: "huggingface" (目前僅支援 HuggingFace)
            dataset_name: 數據集名稱或ID（如 "cifar10", "imagenet-1k", "common_voice" 等）
            destination_path: 如果指定，下載到這個路徑；否則使用默認路徑(self.datasets_tmp_root)
            split: 數據集分割（如 "train", "test", "validation"），若為 None 則下載所有分割
            config_name: 數據集配置名稱（對於有多個配置的數據集）
            extract_multimedia: 是否提取並保存多媒體文件（圖像、音頻、視頻等）
            max_samples: 提取多媒體文件時的最大樣本數量（防止數據集過大）
            
        Returns:
            下載後數據集的本地路徑或錯誤信息
        """
        if destination_path is None:
            dataset_name_replace = dataset_name.replace("/", "_")  # 將 '/' 替換為 '_' 以避免路徑問題
            destination_path = os.path.join(self.datasets_tmp_root, dataset_name_replace)
        else:
            dataset_name_replace = dataset_name.replace("/", "_")
            destination_path = os.path.join(os.path.abspath(destination_path), dataset_name_replace)

        os.makedirs(destination_path, exist_ok=True)
        print(f"Downloading dataset '{dataset_name_replace}' to '{destination_path}'...")

        if dataset_source == "huggingface":
            try:
                # 使用 datasets 庫從 HuggingFace Hub 下載數據集
                dataset_kwargs = {
                    "path": dataset_name,
                    "cache_dir": destination_path,
                    "token": self.hf_token if self.hf_token else None
                }
                
                # 添加可選參數
                if config_name:
                    dataset_kwargs["name"] = config_name
                if split:
                    dataset_kwargs["split"] = split
                    
                # 下載數據集
                dataset = load_dataset(**dataset_kwargs)
                
                # 處理不同類型的數據集
                dataset_info = self._process_and_save_dataset(dataset, destination_path, dataset_name, 
                                                            split, extract_multimedia, max_samples)
                
                # 創建數據集信息文件
                info_file = os.path.join(destination_path, "dataset_info.txt")
                with open(info_file, "w", encoding="utf-8") as f:
                    f.write(f"Dataset Name: {dataset_name}\n")
                    f.write(f"Source: {dataset_source}\n")
                    f.write(f"Config: {config_name or 'default'}\n")
                    f.write(f"Downloaded at: {datetime.datetime.now()}\n")
                    
                    # 添加數據集基本信息
                    if hasattr(dataset, 'info') and dataset.info:
                        f.write(f"Description: {getattr(dataset.info, 'description', 'N/A')}\n")
                        f.write(f"Features: {getattr(dataset.info, 'features', 'N/A')}\n")
                    
                    # 添加處理後的數據集統計信息
                    f.write(f"\n=== Dataset Processing Summary ===\n")
                    f.write(f"Dataset Type: {dataset_info.get('dataset_type', 'unknown')}\n")
                    f.write(f"Total Samples: {dataset_info.get('total_samples', 0)}\n")
                    f.write(f"Multimedia Files: {dataset_info.get('multimedia_count', 0)}\n")
                    f.write(f"Output Formats: {', '.join(dataset_info.get('output_formats', []))}\n")
                    
                    if dataset_info.get('multimedia_stats'):
                        f.write(f"\n=== Multimedia Statistics ===\n")
                        for media_type, stats in dataset_info['multimedia_stats'].items():
                            f.write(f"{media_type.capitalize()}: {stats}\n")
                
                print("HuggingFace dataset downloaded successfully.")
                return destination_path
                
            except Exception as e:
                error_msg = f"Error downloading dataset from HuggingFace: {e}"
                print(error_msg)
                return error_msg
                
        else:
            try:
                raise ValueError("Unsupported dataset source. Currently only 'huggingface' is supported.")
            except ValueError as ve:
                return f"Error: {ve}"

    def upload_to_lakefs(self, repo_name: str, local_path: str, branch_name: Optional[str] = "main", 
                        commit_message: Optional[str] = None) -> str:
        """
        將本地數據集資料夾上傳至 lakeFS 儲存庫的指定分支。
        
        Args:
            repo_name (str): lakeFS 儲存庫名稱(需符合格式規範：只允許小寫字母、數字、"-")
            local_path (str): 包含數據集的本地資料夾路徑
            branch_name (Optional[str]): 分支名稱，默認為 "main"
            commit_message (Optional[str]): 本次上傳的提交訊息。若為 None，則會自動生成
            
        Returns:
            str: lakefs commit URI 或錯誤信息
        """
        repo = self._ensure_lakefs_repo(repo_name, branch_name)
        if repo is None:
            return f"Failed to ensure lakeFS repository: {repo_name}."
            
        # 上傳數據集    
        try:
            print(f"Uploading dataset from '{local_path}' to lakeFS repo '{repo_name}' on branch '{branch_name}'...")
            branch_to_upload = lakefs.repository(repo_name, client=self.lakefs_client).branch(branch_name)
            dataset_source_path = Path(local_path).expanduser().resolve()
            folder_name = dataset_source_path.name
            
            # 遞歸上傳所有檔案
            all_files = dataset_source_path.rglob('*')
            for file_path in all_files:
                if file_path.is_file():
                    relative_path = file_path.relative_to(dataset_source_path)
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
            message = commit_message or f"Upload dataset {folder_name} to {branch_name}"
            commit = branch_to_upload.commit(message)
            print(f"Dataset uploaded and committed to lakeFS. Commit ID: {commit.id}")
            return f"lakefs://{repo_name}/{commit.id}/"
            
        except Exception as e:
            print(f"Error committing changes to lakeFS: {e}")
            return "Failed to commit changes to lakeFS."
    
    def list_lakefs_repos(self) -> list:
        """
        列出所有 lakeFS 儲存庫中屬於數據集的儲存庫
        
        Returns:
            list: 儲存庫資訊列表，格式如下：         
            [{
                "repo_id": "my-dataset-repo",
                "repo_creation_date": "2023-10-01 12:00:00",
                "storage_namespace": "s3://my-bucket/datasets/my-dataset-repo",
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
                repo = repo.properties
                
                # 只處理數據集相關的儲存庫
                if not repo.storage_namespace.startswith(self.lakefs_datasets_storage_namespace):
                    continue
                    
                branches = []
                for branch in lakefs.repository(repo.id, client=self.lakefs_client).branches():
                    creation_date = datetime.datetime.fromtimestamp(
                        branch.get_commit().creation_date
                    ).strftime('%Y-%m-%d %H:%M:%S')
                    branches.append({
                        "branch_id": branch.id,
                        "latest_commit_id": branch.get_commit().id,
                        "latest_commit_time": creation_date,
                        "committer": branch.get_commit().committer,
                        "commit_message": branch.get_commit().message,
                    })
                    
                repo_creation_date = datetime.datetime.fromtimestamp(
                    repo.creation_date
                ).strftime('%Y-%m-%d %H:%M:%S')
                
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
        從 lakeFS 儲存庫的指定分支下載數據集到本地。
        
        Args:
            repo_name (str): lakeFS 儲存庫名稱
            download_name (Optional[str]): 下載後的本地資料夾名稱，若為 None 則使用 repo_name
            commit_id (Optional[str]): 指定的 commit ID，若為 None 則使用 branch_name 的最新 commit
            branch_name (Optional[str]): 分支名稱，默認為 None，通常不需填
            destination_path_root (Optional[str]): 本地下載路徑，若為 None 則使用 self.download_from_lakefs_root
            
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
            if branch_name is None:
                branch_name = "main"
            try:
                commit_id = lakefs.repository(repo_name, client=self.lakefs_client).branch(branch_name).get_commit().id
                print(f"Using latest commit ID from branch '{branch_name}': {commit_id}")
            except Exception as e:
                error_msg = f"Failed to fetch latest commit ID from branch '{branch_name}': {e}"
                print(error_msg)
                return error_msg
                
        destination_path = os.path.join(destination_path_root, download_name, commit_id)
        if os.path.exists(destination_path) and os.listdir(destination_path):
            print(f"Dataset already exists in local cache: {destination_path}")
            return destination_path
        os.makedirs(destination_path, exist_ok=True)

        print(f"Downloading dataset from lakeFS repo '{repo_name}' with commit_id:'{commit_id}' to '{destination_path}'...")
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
                print(f"Downloaded: {obj.path} -> {obj_path}")
            except Exception as e:
                print(f"下載失敗: {obj.path}, 錯誤: {e}")
        
        print(f"Dataset download completed. Local path: {destination_path}")
        return destination_path

    def list_local_datasets(self, from_lakefs: Optional[bool] = False) -> list:
        """
        從本地文件系統掃描數據集
        
        Args:
            from_lakefs (Optional[bool]): 是否掃描從 lakeFS 下載的數據集目錄，默認為 False
            
        Returns:
            list: 數據集列表，格式如下：
            [
                {
                    'name': 'dataset_name',
                    'path': '/path/to/dataset',
                    'size_mb': 1024.5,
                    'file_count': 100
                },
                ...
            ]
        """
        datasets = []
        scan_root = self.download_from_lakefs_root if from_lakefs else self.datasets_tmp_root
        
        try:
            for item in os.listdir(scan_root):
                item_path = os.path.join(scan_root, item)
                if os.path.isdir(item_path):
                    # 計算目錄大小和文件數量
                    total_size = 0
                    file_count = 0
                    for root, dirs, files in os.walk(item_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                total_size += os.path.getsize(file_path)
                                file_count += 1
                    
                    size_mb = round(total_size / (1024 * 1024), 2)
                    datasets.append({
                        "name": item,
                        "path": item_path,
                        "size_mb": size_mb,
                        "file_count": file_count
                    })
            return datasets
            
        except Exception as e:
            print(f"Error listing local datasets: {e}")
            return [{"error": str(e)}]

    def register_lakefs_to_mlflow(self, 
                                  registered_name: str,
                                  lakefs_repo: str,
                                  dataset_type: str,
                                  description: Optional[str] = "",
                                  commit_id: Optional[str] = None,
                                  lakefs_branch: Optional[str] = "main",
                                  tags: Optional[Dict[str, str]] = None) -> dict:
        """
        將 lakeFS 中已提交的數據集版本註冊到 MLflow 中。

        Args:
            registered_name (str): 在 MLflow 中註冊的數據集名稱
            lakefs_repo (str): lakeFS 儲存庫名稱
            dataset_type (str): 數據集類型，如 "training", "validation", "test", "raw"
            description (Optional[str]): 數據集描述信息
            commit_id (Optional[str]): lakeFS 中的 commit ID，默認使用最新 commit
            lakefs_branch (Optional[str]): lakeFS 分支名稱，默認為 "main"
            tags (Optional[Dict[str, str]]): 額外的標籤信息
            
        Returns:
            Dict: 註冊的數據集版本資訊或錯誤信息
        """
        if commit_id is None:
            try:
                commit_id = lakefs.repository(lakefs_repo, client=self.lakefs_client).branch(lakefs_branch).get_commit().id
                print(f"Using latest commit ID from branch '{lakefs_branch}': {commit_id}")
            except Exception as e:
                error_msg = f"Failed to fetch latest commit ID from branch '{lakefs_branch}': {e}"
                print(error_msg)
                return {"error": error_msg}

        dataset_uri = f"lakefs://{lakefs_repo}/{commit_id}/"
        
        # 建立實驗
        mlflow.set_experiment(experiment_name=f"datasets_{registered_name}")
        
        with mlflow.start_run(run_name=f"Register-{registered_name}", description=description) as run:
            run_id = run.info.run_id
            
            # 創建虛擬 artifact 來滿足 MLflow 要求
            dummy_dataset_path = os.path.join(self.datasets_tmp_root, f"dummy_for_{run_id}")
            os.makedirs(dummy_dataset_path, exist_ok=True)
            
            readme_content = (
                f"This MLflow dataset version is a pointer to a dataset stored in lakeFS.\n"
                f"Physical Location: {dataset_uri}\n"
                f"Repository: {lakefs_repo}\n"
                f"Commit ID: {commit_id}\n"
                f"Dataset Type: {dataset_type}\n"
            )
            
            with open(os.path.join(dummy_dataset_path, "dataset_pointer.md"), "w") as f:
                f.write(readme_content)
            
            try:
                mlflow.log_artifact(str(dummy_dataset_path), artifact_path="dataset_pointer")
            except Exception as e:
                log_error = f"Error logging dummy artifact to MLflow: {e}"
                print(log_error)
                shutil.rmtree(dummy_dataset_path, ignore_errors=True)
                return {"error": log_error}
                
            shutil.rmtree(dummy_dataset_path, ignore_errors=True)
            
            # 準備標籤
            dataset_tags = {
                "physical_path": dataset_uri,
                "dataset_type": dataset_type,
                "lakefs_repo": lakefs_repo,
                "lakefs_commit_id": commit_id,
                "lakefs_branch": lakefs_branch
            }
            
            if tags:
                dataset_tags.update(tags)
            
            # 記錄數據集
            try:
                # 使用 MLflow 的數據集記錄功能
                mlflow.log_params({
                    "dataset_name": registered_name,
                    "dataset_type": dataset_type,
                    "lakefs_repo": lakefs_repo,
                    "commit_id": commit_id
                })
                
                # 設置標籤
                for key, value in dataset_tags.items():
                    mlflow.set_tag(key, value)
                
                print(f"Dataset '{registered_name}' registered successfully in MLflow.")
                print(f"  - MLflow Run ID: {run_id}")
                print(f"  - LakeFS Commit ID: {commit_id}")
                
                return {
                    "name": registered_name,
                    "run_id": run_id,
                    "physical_path": dataset_uri,
                    "lakefs_commit_id": commit_id,
                    "dataset_type": dataset_type,
                    "tags": dataset_tags
                }
                
            except Exception as e:
                register_error = f"Error registering dataset to MLflow: {e}"
                print(register_error)
                return {"error": register_error}

    def list_mlflow_datasets(self, experiment_name_prefix: Optional[str] = "datasets_") -> list:
        """
        從 MLflow 獲取已註冊的數據集資訊。
        
        Args:
            experiment_name_prefix (Optional[str]): 實驗名稱前綴，用於篩選數據集實驗
            
        Returns:
            list: 包含數據集詳細資訊的列表
        """
        datasets = []
        try:
            experiments = self.mlflow_client.search_experiments()
            
            for experiment in experiments:
                if experiment_name_prefix and not experiment.name.startswith(experiment_name_prefix):
                    continue
                    
                runs = self.mlflow_client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"]
                )
                
                for run in runs:
                    dataset_info = {
                        "experiment_name": experiment.name,
                        "run_id": run.info.run_id,
                        "dataset_name": run.data.params.get("dataset_name", "unknown"),
                        "dataset_type": run.data.params.get("dataset_type", "unknown"),
                        "lakefs_repo": run.data.params.get("lakefs_repo", ""),
                        "commit_id": run.data.params.get("commit_id", ""),
                        "physical_path": run.data.tags.get("physical_path", ""),
                        "start_time": datetime.datetime.fromtimestamp(
                            run.info.start_time / 1000
                        ).strftime('%Y-%m-%d %H:%M:%S'),
                        "status": run.info.status,
                        "tags": run.data.tags
                    }
                    datasets.append(dataset_info)
                    
            return datasets
            
        except Exception as e:
            error_msg = f"Error listing MLflow datasets: {e}"
            print(error_msg)
            return [{"error": error_msg}]

    def get_dataset_details_from_mlflow(self, dataset_name: str) -> dict:
        """
        根據數據集名稱獲取詳細信息。
        
        Args:
            dataset_name (str): 數據集名稱
            
        Returns:
            dict: 包含數據集詳細資訊的字典，如果找不到則返回錯誤訊息
        """
        try:
            all_datasets = self.list_mlflow_datasets()
            
            for dataset in all_datasets:
                if dataset.get("dataset_name") == dataset_name:
                    return dataset
                    
            return {"error": f"Dataset '{dataset_name}' not found in MLflow."}
            
        except Exception as e:
            error_msg = f"Error fetching dataset details for '{dataset_name}': {e}"
            print(error_msg)
            return {"error": error_msg}

    def download_by_mlflow_uri(self, mlflow_run_id: str, 
                              destination_root: Optional[str] = None) -> str:
        """
        根據 MLflow run ID 下載對應的數據集。
        
        Args:
            mlflow_run_id (str): MLflow run ID
            destination_root (Optional[str]): 本地下載根目錄
            
        Returns:
            str: 下載後的本地路徑或錯誤信息
        """
        try:
            run = self.mlflow_client.get_run(mlflow_run_id)
            
            lakefs_repo = run.data.params.get("lakefs_repo")
            commit_id = run.data.params.get("commit_id")
            dataset_name = run.data.params.get("dataset_name", "unknown")
            
            if not lakefs_repo or not commit_id:
                return "Error: Missing lakeFS repository or commit ID in MLflow run."
            
            return self.download_from_lakefs(
                repo_name=lakefs_repo,
                download_name=dataset_name,
                commit_id=commit_id,
                destination_path_root=destination_root
            )
            
        except Exception as e:
            error_msg = f"Error downloading dataset by MLflow URI: {e}"
            print(error_msg)
            return error_msg

    def validate_dataset_structure(self, dataset_path: str) -> dict:
        """
        驗證數據集結構的完整性。
        
        Args:
            dataset_path (str): 數據集路徑
            
        Returns:
            dict: 驗證結果，包含文件統計和可能的問題
        """
        validation_result = {
            "valid": True,
            "total_files": 0,
            "total_size_mb": 0,
            "file_types": {},
            "warnings": [],
            "errors": []
        }
        
        try:
            if not os.path.exists(dataset_path):
                validation_result["valid"] = False
                validation_result["errors"].append(f"Dataset path does not exist: {dataset_path}")
                return validation_result
            
            total_size = 0
            file_count = 0
            file_types = {}
            
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        file_size = os.path.getsize(file_path)
                        total_size += file_size
                        file_count += 1
                        
                        # 統計文件類型
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext:
                            file_types[file_ext] = file_types.get(file_ext, 0) + 1
                        else:
                            file_types["no_extension"] = file_types.get("no_extension", 0) + 1
            
            validation_result["total_files"] = file_count
            validation_result["total_size_mb"] = round(total_size / (1024 * 1024), 2)
            validation_result["file_types"] = file_types
            
            # 檢查常見問題
            if file_count == 0:
                validation_result["valid"] = False
                validation_result["errors"].append("Dataset is empty")
            
            if total_size == 0:
                validation_result["warnings"].append("All files are empty")
            
            return validation_result
            
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {e}")
            return validation_result

# 全域實例
dataset_manager = DatasetManager()