#!/usr/bin/env bash
set -euo pipefail
docker save topaneu-task1-sanity:v0.1.0 | gzip -c > topaneu-task1-sanity-v0.1.0.tar.gz
