# PetMask

U-Net 기반 반려동물 세그멘테이션 및 배포 프로젝트입니다.  
사용자가 이미지를 업로드하면 AI 모델이 반려동물 영역을 분리하여 segmentation mask를 생성합니다.

---

## Project Overview

이 프로젝트는 다음 과정을 직접 구현하는 것을 목표로 합니다.

- U-Net 기반 Semantic Segmentation
- PyTorch 기반 모델 학습 및 추론
- FastAPI 기반 Inference Server 구축
- Docker 기반 컨테이너화 및 배포

---

## Tech Stack

### AI / Deep Learning
- PyTorch
- Torchvision
- U-Net
- OpenCV
- NumPy

### Backend
- FastAPI

### Frontend
- CSS
- HTML
- Java Script

### Deployment
- Docker

---

## Dataset

- Oxford-IIIT Pet Dataset

---

## Features

- Pet Segmentation Inference
- Binary Mask Generation
- Overlay Visualization
- REST API Inference
- Docker-based Deployment

---

## Project Structure

```text
petmask/
├── src/                 # 모델 구조, 학습 및 평가 코드
├── backend/             # FastAPI 기반 백엔드 및 모델 추론 API
├── frontend/            # HTML/CSS/JavaScript 기반 웹 UI
├── checkpoints/         # 학습된 모델 체크포인트
├── temp/                # 추론 결과 이미지 임시 저장
├── Dockerfile           # Docker 이미지 빌드 설정
├── pyproject.toml       # Python 프로젝트 및 의존성 설정
├── uv.lock              # uv 의존성 버전 고정
└── README.md            # 프로젝트 설명 및 실행 방법