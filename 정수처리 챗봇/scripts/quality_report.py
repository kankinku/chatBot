#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
주간 품질 리포트 생성 스크립트(스켈레톤)
- GOLDENSET_PATH 환경변수로 골든셋 지정
- 서버 /metrics, /config/info에서 지표/버전 조회
"""

import os
import json
import urllib.request

try:
    from core.config.unified_config import get_config
except Exception:
    import os
    def get_config(key, default=None):
        return os.getenv(key, default)
API_BASE = get_config('API_BASE', 'http://localhost:8008')


def http_get(path: str):
    with urllib.request.urlopen(API_BASE + path) as resp:
        return json.loads(resp.read().decode('utf-8'))


def evaluate_goldenset(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {"items": len(data)}
    except Exception as e:
        return {"error": str(e)}


def render_template(metrics: dict, config: dict):
    tpl_path = os.path.join('docs', 'QUALITY_REPORT_TEMPLATE.md')
    if not os.path.exists(tpl_path):
        return None
    with open(tpl_path, 'r', encoding='utf-8') as f:
        tpl = f.read()
    m = metrics.get('metrics', {})
    repl = {
        '{{start}}': '-',
        '{{end}}': '-',
        '{{config_version}}': str(config.get('config_version')),
        '{{ndcg10}}': '-',
        '{{mrr10}}': '-',
        '{{recallk}}': '-',
        '{{acc}}': '-',
        '{{p95}}': str(m.get('rolling', {}).get('p95_ms', '0')),
        '{{errors5m}}': str(m.get('rolling', {}).get('errors_5m', '0')),
        '{{config_diff}}': json.dumps({}, ensure_ascii=False)
    }
    for k, v in repl.items():
        tpl = tpl.replace(k, v)
    return tpl


def main():
    metrics = http_get('/metrics')
    config = http_get('/config/info')
    report = {
        'config_version': config.get('config_version'),
        'metrics': metrics.get('metrics', {}),
    }
    gs_path = get_config('GOLDENSET_PATH')
    if gs_path:
        report['goldenset'] = evaluate_goldenset(gs_path)
    out = 'weekly_quality_report.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f'report saved to {out}')
    rendered = render_template(metrics, config)
    if rendered:
        out_md = 'weekly_quality_report.md'
        with open(out_md, 'w', encoding='utf-8') as f:
            f.write(rendered)
        print(f'markdown saved to {out_md}')


if __name__ == '__main__':
    main()
