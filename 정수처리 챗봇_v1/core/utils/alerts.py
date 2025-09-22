"""
간단한 알림 유틸: Slack/메일 스텁. 환경/SSOT와 연계해 임계 초과시 호출.
"""

from typing import Optional
import os

try:
    from core.config.unified_config import get_config
except Exception:
    def get_config(key, default=None):
        return os.getenv(key, default)


def notify_slack(text: str) -> bool:
    webhook = get_config("SLACK_WEBHOOK_URL", "")
    if not webhook:
        return False
    try:
        import requests
        resp = requests.post(webhook, json={"text": text}, timeout=5)
        return resp.status_code < 300
    except Exception:
        return False


def notify_email(subject: str, body: str) -> bool:
    # 단순 스텁: 실제 메일 연동은 SMTP/서비스 필요
    return False


