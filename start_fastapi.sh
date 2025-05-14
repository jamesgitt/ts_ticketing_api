#!/bin/bash
cd /home/ubuntu/ts_ticketing_api
source llmvenv/bin/activate
exec uvicorn rest_api:app --host 0.0.0.0 --port 8000
