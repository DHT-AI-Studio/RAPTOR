# RAPTOR 0.2 Version
**說明：**  
RAPTOR 0.2 Version 在 0.1 版本的基礎上新增以下內容：
1. 使用 **Keycloak** 提供基於角色的存取控制（英語：Role-based access control，**RBAC**）
2. 使用 **Qwen** 建構的時序推理引擎提供影片分析服務
3. 


## Project Structure
```bash
.
├── CIE_System_Design_and_Architecture_1.8.pdf
├── doc
│   ├── Aigle_0.1_data.svg
│   ├── Aigle_0.1_system.svg
│   └── CIE_System_Technical_Implementation_1.2.pdf
├── raptor                                                                  # raptor source code
│   ├── AiModelLifecycle
│   ├── Api_Gateway                                                         # 存取 raptor 服務的 API source code
│   ├── asset_management
│   ├── check-services.sh
│   ├── DEPLOYMENT.md
│   ├── deploy.sh
│   ├── docker-compose.yaml
│   ├── kafka
│   ├── logs.sh
│   ├── qdrant_search_docker
│   ├── README.md
│   ├── Redis
│   └── Temporal_Reasoning_Engine
├── README_0.2.md                                                           # this file
├── README.md                                                               # 0.1 version README.md
├── requirements.txt                                                        # python dependency
└── test_file                                                               # 提供測試用各種類型文件目錄
    ├── 2p-01.wav
    ├── 3.服務建議書徵求須知(RFP).pdf
    ├── AI02.wav
    ├── AI.API.v2.0 (1).docx
    ├── _E7_99_BC_E7_A5_A8.jpg
    ├── EF25Y01.csv
    ├── jogging.mp4
    └── 【魔法壞女巫 ： 第二部】幕後花絮 - 11月21日 全台戲院見 [63J7rmWlCwE].mp4
```

## 