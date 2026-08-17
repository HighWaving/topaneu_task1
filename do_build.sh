#!/usr/bin/env bash
set -euo pipefail
docker build --platform=linux/amd64 --tag topaneu-task1-sanity:v0.1.0 .
