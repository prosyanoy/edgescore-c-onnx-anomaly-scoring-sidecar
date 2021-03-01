#!/usr/bin/env bash
# Build the image, create a kind cluster, side-load the image and deploy the sidecar demo.
set -euo pipefail
cd "$(dirname "$0")/.."

for tool in docker kind kubectl; do
  command -v "$tool" >/dev/null || { echo "missing required tool: $tool" >&2; exit 1; }
done

docker build -t edgescore:latest .
kind get clusters | grep -qx edgescore || kind create cluster --config k8s/kind-config.yaml
kind load docker-image edgescore:latest --name edgescore
kubectl --context kind-edgescore apply -k k8s/
kubectl --context kind-edgescore -n edgescore-demo rollout status deploy/demo-app --timeout=180s

cat <<MSG

Deployed. Try:
  kubectl --context kind-edgescore -n edgescore-demo port-forward deploy/demo-app 8088:8088 &
  curl -s localhost:8088/metrics | grep -E 'edgescore_(events_scored_total|anomalies_total|latency_p)'
  curl -s 'localhost:8088/v1/results?anomalies=1&limit=5'
  kubectl --context kind-edgescore -n edgescore-demo logs deploy/demo-app -c app --tail=5
Tear down with: kind delete cluster --name edgescore
MSG
