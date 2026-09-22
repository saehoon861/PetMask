# --- Builder 스테이지 ---
FROM python:3.11-slim AS builder

WORKDIR /app

# 빌드에 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
RUN pip install --no-cache-dir uv

# 의존성 파일 복사 및 설치
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache

# --- Runner 스테이지 ---
FROM python:3.11-slim

WORKDIR /app

# 런타임에 필요한 최소한의 시스템 패키지만 설치 (필요시 추가)
# RUN apt-get update && apt-get install -y --no-install-recommends ... && rm -rf /var/lib/apt/lists/*

# Builder 스테이지에서 설치된 파이썬 패키지 및 환경 복사
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 소스 코드 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 모델 경로 환경 변수 설정
ENV MODEL_CHECKPOINT=/app/checkpoints/checkpoint.pth

# 애플리케이션 실행 명령어
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
