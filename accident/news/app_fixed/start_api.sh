#!/usr/bin/env bash
# Run this in its own terminal window and leave it running.
cd "$(dirname "$0")/api" || exit 1
python app.py
