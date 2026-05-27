# PetMask

U-Net 기반 반려동물 세그멘테이션 및 배포 프로젝트입니다.  
사용자가 이미지를 업로드하면 AI 모델이 반려동물 영역을 분리하여 segmentation mask를 생성합니다.

---

## Project Overview

이 프로젝트는 다음 과정을 직접 구현하는 것을 목표로 합니다.

- U-Net 기반 Semantic Segmentation
- PyTorch 기반 모델 학습 및 추론
- FastAPI 기반 Inference Server 구축
- React(Vite) 기반 Demo Frontend 구현
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
- React
- Vite

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
├── src/                 # 모델 학습 코드
├── app/                 # FastAPI 서버
├── frontend/            # React 프론트엔드
├── weights/             # 학습된 모델 가중치
├── outputs/             # 예측 결과 저장
├── Dockerfile
└── README.md