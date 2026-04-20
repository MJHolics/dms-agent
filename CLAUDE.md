# Driver Monitoring System (DMS) Agent — 프로젝트 컨텍스트

## 목표
LangGraph 기반 운전자 상태 모니터링 에이전트.
웹캠/영상 입력 → 다중 에이전트 협력 → 졸음/부주의 감지 + 경고 출력.

## 사용자 배경
- LLM/RAG/PEFT 경험 (강점)
- CV 경험: YOLO/SAM/DepthAnything, 자율주행 파이프라인
- FastAPI + Docker 경험
- LangGraph 에이전트 프레임워크 경험
- 목표: 현대차/모빌리티 AI 취업 포트폴리오

## 기술 스택
- **MediaPipe Face Mesh** — 눈/입/고개 추적
- **YOLOv8** — 휴대폰 등 객체 감지
- **LangGraph** — 에이전트 오케스트레이션
- **Ollama (Qwen2.5-VL 7B)** — 로컬 LLM (무료, 무제한)
- **FastAPI + Docker** — 서빙

## 에이전트 구조
```
웹캠/영상 입력
      ↓
Frame Processor
      ↓
Face Analysis Agent       Object Detection Agent
(EAR/MAR/Head Pose)      (YOLO - 폰, 담배 등)
      ↓                         ↓
      └──── State Classifier Agent ────┘
                    ↓
           LLM Reasoning Agent (Ollama)
           상황 판단 + 경고 메시지 생성
                    ↓
           Alert Manager Agent
           경고 레벨 1 / 2 / 3 출력
```

## 4단계 로드맵

### Notebook 01 — MediaPipe + 눈/입 추적 기초
- MediaPipe Face Mesh 설치 및 기본 사용법
- EAR (Eye Aspect Ratio) — 눈 감김 감지
- MAR (Mouth Aspect Ratio) — 하품 감지
- Head Pose Estimation — 고개 방향 감지
- 웹캠 실시간 테스트

### Notebook 02 — 졸음/부주의 감지 모듈
- EAR 임계값 기반 졸음 판단
- PERCLOS 알고리즘 구현
- YOLOv8로 휴대폰/담배 감지
- 멀티 신호 통합 (눈 + 입 + 고개 + 객체)

### Notebook 03 — LangGraph 에이전트 구성
- Face Analysis Agent 구현
- Object Detection Agent 구현
- State Classifier Agent 구현
- Ollama 연동 LLM Reasoning Agent
- Alert Manager Agent

### Notebook 04 — 풀 파이프라인 통합
- 전체 에이전트 파이프라인 연결
- 실시간 영상 처리
- 경고 레벨 시스템 (정상/주의/위험)
- FastAPI 서빙

## 환경
- Python: Anaconda
- LLM: Ollama 로컬 (Qwen2.5-VL 7B 또는 Llama3.2-Vision 11B)
- GPU: RTX 4080 Super (16GB VRAM)
- API 키: 불필요 (완전 로컬)

## 노트북 작성 규칙
- 한글 폰트 설정 (첫 셀):
  ```python
  import matplotlib
  matplotlib.rcParams['font.family'] = 'Malgun Gothic'
  matplotlib.rcParams['axes.unicode_minus'] = False
  ```
- 각 셀마다 목적 주석 포함

## 핵심 알고리즘
- **EAR** = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) → 0.25 이하면 눈 감김
- **MAR** = (|p2-p8| + |p3-p7| + |p4-p6|) / (2 * |p1-p5|) → 0.6 이상이면 하품
- **PERCLOS** = 단위 시간 내 눈 감긴 프레임 비율 → 0.15 이상이면 졸음

## 진행 상황
- [x] 프로젝트 구조 생성
- [ ] Notebook 01 — MediaPipe 기초
- [ ] Notebook 02 — 졸음/부주의 감지
- [ ] Notebook 03 — LangGraph 에이전트
- [ ] Notebook 04 — 풀 파이프라인 통합
- [ ] FastAPI 서빙
- [ ] Docker 배포
