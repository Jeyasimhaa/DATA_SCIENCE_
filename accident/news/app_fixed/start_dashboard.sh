#!/usr/bin/env bash
# Run this in a SECOND terminal window (after start_api.sh is running).
cd "$(dirname "$0")/dashboard" || exit 1
streamlit run app.py
