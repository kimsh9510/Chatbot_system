# Chatbot
환자 낙상 감지 이벤트와 환자 정보를 기반으로 상황 판단 및 대처 방안을 제공하는 Chatbot
Retrieval-Augmented Generation (RAG)을 기반으로 적절한 대응 시나리오 제공

## Installation
```bash
git clone https://github.com/kimsh9510/Chatbot_system.git
cd Disaster_Chatbot
```

### 가상 환경 설정
- 개인 PC 새 가상 환경 생성
```bash
conda create -n chatbot_env python=3.12
conda activate chatbot_env
```

- 가상 환경 내 라이브러리 설치
```bash
pip install -r requirements.txt
```

### Execution Requirments
프로젝트를 정상적으로 실행하기 위해서는 **75번 서버 내의 파일을 별도로 복사** 해야함
- 폴더 경로 : Documents/Seonhyeong/Disaster_Chatbot
- 필수 항목
    - `Dataset/` : 환자 정보가 포함된 파일
    -  `.env` : LLM 모델의 API Key
- 프로그램 실행
```bash
python main.py
```

