# syntax=docker/dockerfile:1
# Stage 1: train the model and export it to ONNX (reproducible: fixed seed, synthetic data).
FROM python:3.12-slim AS model
WORKDIR /src/training
COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY training/ .
RUN python train.py --out /out

# Stage 2: build the C++ service against the prebuilt ONNX Runtime CPU release.
FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      g++ cmake make ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt .
COPY include include
COPY src src
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DEDGESCORE_BUILD_TESTS=OFF -DEDGESCORE_BUILD_BENCH=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/edgescore \
 && cmake --build build -j"$(nproc)" && cmake --install build

# Stage 3: minimal runtime, non-root.
FROM ubuntu:24.04
RUN useradd --system --uid 10001 --no-create-home edgescore
COPY --from=build /opt/edgescore /opt/edgescore
COPY --from=model /out /opt/edgescore/models
ENV EDGESCORE_MODEL=/opt/edgescore/models/model.onnx \
    EDGESCORE_PORT=8088
USER 10001
EXPOSE 8088
ENTRYPOINT ["/opt/edgescore/bin/edgescore"]
