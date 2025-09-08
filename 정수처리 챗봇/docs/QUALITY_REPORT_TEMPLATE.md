# 품질 리포트 템플릿

- 리포트 기간: {{start}} ~ {{end}}
- 시스템 버전: {{config_version}}

## 요약
- nDCG@10: {{ndcg10}}
- MRR@10: {{mrr10}}
- Recall@k: {{recallk}}
- 정답률: {{acc}}
- p95(ms): {{p95}}
- 오류율(5m): {{errors5m}}

## 변경점
- 설정 변경(diff 요약):
```
{{config_diff}}
```

## 세부 결과
- 도메인별 세부 표 (생략)

## 조치 사항
- 개선 항목:
- 리스크:

## 로그 품질 체크리스트 (SSOT 적용)
- 운영 로그는 `utils/unified_logger.py` 사용 여부 확인
- 세션ID 전파율 점검(`unified_logger.set_session_id`)