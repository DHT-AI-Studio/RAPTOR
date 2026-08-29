#!/bin/bash
cd "$(dirname "$0")/deployment/modules" && python build.py "$@"
