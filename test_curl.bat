@echo off
curl -s --max-time 180 -w "HTTP_STATUS:%%{http_code}" http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d @test_request.json
