#!/bin/bash
# 더블클릭: 1단계가 끝나길 기다렸다가 2단계(데이터 조합 비교) 자동 실행
bash "$(dirname "$0")/phase2.sh"
