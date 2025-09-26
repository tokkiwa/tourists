#!/usr/bin/env bash
# exit on error
set -o errexit

pip install poetry

poetry install --no-root --no-dev