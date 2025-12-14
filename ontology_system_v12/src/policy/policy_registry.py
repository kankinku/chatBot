"""
Simple policy registry for managing active/variant policies.

초기 구현은 JSONL 파일 기반으로 구성하여 외부 의존성을 최소화한다.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from src.policy.contracts import PolicyBundle

logger = logging.getLogger(__name__)


class PolicyRegistry:
    """
    정책 저장/조회/승격을 담당하는 레지스트리.
    """

    def __init__(self, storage_path: str = "data/policies/policies.jsonl"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_policy_id: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def _append(self, bundle: PolicyBundle, metadata: Optional[Dict[str, Any]] = None) -> None:
        record = {
            "policy_id": bundle.policy_id,
            "version": bundle.version,
            "created_at": bundle.created_at.isoformat(),
            "params": bundle.params,
            "metadata": metadata or {},
        }
        with self.storage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Saved policy {bundle.policy_id} (v{bundle.version})")

    def _load_all(self) -> Dict[str, PolicyBundle]:
        bundles: Dict[str, PolicyBundle] = {}
        if not self.storage_path.exists():
            return bundles
        with self.storage_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    bundle = PolicyBundle(
                        policy_id=rec["policy_id"],
                        version=int(rec.get("version", 1)),
                        created_at=datetime.fromisoformat(rec["created_at"]),
                        params=rec.get("params", {}),
                    )
                    bundles[bundle.policy_id] = bundle
                except Exception as e:
                    logger.warning(f"Failed to load policy line: {e}")
        return bundles

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def register_policy(self, bundle: PolicyBundle) -> None:
        """
        Manually register a policy bundle (e.g. from optimizer).
        """
        self._append(bundle)

    def get_policy(self, policy_id: str) -> Optional[PolicyBundle]:
        return self._load_all().get(policy_id)

    def get_active_policy(self) -> PolicyBundle:
        bundles = self._load_all()
        if self._active_policy_id and self._active_policy_id in bundles:
            return bundles[self._active_policy_id]

        # fallback: latest policy or default
        if bundles:
            latest = sorted(bundles.values(), key=lambda b: (b.version, b.created_at))[-1]
            self._active_policy_id = latest.policy_id
            return latest

        # create default if none exists
        default = PolicyBundle(
            policy_id="policy_default",
            version=1,
            params={
                "fusion": {
                    "domain_weight": 0.7,
                    "personal_weight": 0.3,
                    "semantic_penalty": 0.0,
                    "decay_lambda": 0.1,
                },
                "routing": {
                    "domain_threshold": 0.55,
                    "personal_threshold": 0.35,
                },
                "retrieval": {
                    "max_depth": 4,
                    "max_paths": 10,
                },
            },
        )
        self._append(default, metadata={"auto_created": True})
        self._active_policy_id = default.policy_id
        return default

    def set_active_policy(self, policy_id: str) -> None:
        bundles = self._load_all()
        if policy_id not in bundles:
            raise ValueError(f"Policy {policy_id} not found")
        self._active_policy_id = policy_id
        self._active_policy_id = policy_id
        logger.info(f"Active policy set to {policy_id}")

    def activate_policy(self, policy_id: str) -> None:
        self.set_active_policy(policy_id)

    def save_policy_variant(
        self,
        parent_id: str,
        params: Dict[str, Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        parent = self.get_policy(parent_id) or self.get_active_policy()
        new_version = parent.version + 1
        new_bundle = PolicyBundle(
            version=new_version,
            params=params,
        )
        self._append(new_bundle, metadata={"parent": parent.policy_id, "metrics": metrics or {}})
        return new_bundle.policy_id


# Singleton-style helper
_registry: Optional[PolicyRegistry] = None


def get_policy_registry() -> PolicyRegistry:
    global _registry
    if _registry is None:
        _registry = PolicyRegistry()
    return _registry
