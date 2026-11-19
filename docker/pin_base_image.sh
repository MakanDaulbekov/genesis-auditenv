#!/bin/sh
# Закрепить базовый образ по digest: пишет docker/base-image.txt (коммитить вместе с Dockerfile).
set -eu
TAG="${1:-python:3.11-slim}"
docker pull "$TAG" >/dev/null
DIGEST=$(docker inspect --format '{{index .RepoDigests 0}}' "$TAG")
echo "$DIGEST" > "$(dirname "$0")/base-image.txt"
echo "закреплено: $DIGEST"
