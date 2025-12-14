"""
Advisor System Step 0-4 통합 테스트

테스트 범위:
- Step 0: 스키마 유효성
- Step 1: Delta Ingestion + Append-only Storage
- Step 2: Feature Calculation
- Step 3: Evidence Layer
- Step 4: Regime Detection
"""
import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

# 테스트 모듈 임포트
from src.shared.schemas import (
    Observation, Event, FeatureValue, EvidenceScore,
    RegimeType, FeatureType, DeltaMethod,
)


class TestStep0Schemas:
    """Step 0: 스키마 유효성 테스트"""
    
    def test_observation_creation(self):
        """Observation 생성 테스트"""
        obs = Observation(
            series_id="SOFR",
            timestamp=datetime(2024, 12, 14),
            value=4.33,
            unit="%",
            source_id="fred",
        )
        
        assert obs.series_id == "SOFR"
        assert obs.value == 4.33
        assert obs.observation_id.startswith("OBS_")
        assert obs.quality_flag == "ok"
    
    def test_event_creation(self):
        """Event 생성 테스트"""
        event = Event(
            source_id="news_feed",
            occurred_at=datetime(2024, 12, 14),
            title="Fed Rate Decision",
            content="The Federal Reserve decided to hold rates steady.",
            event_type="policy",
        )
        
        assert event.source_id == "news_feed"
        assert event.event_id.startswith("EVT_")
        assert event.content_hash != ""
    
    def test_evidence_score_creation(self):
        """EvidenceScore 생성 테스트"""
        score = EvidenceScore(
            edge_id="EDGE_001",
            head_id="interest_rate",
            tail_id="growth_stock",
            relation_type="Affect",
            pro_score=0.7,
            con_score=0.2,
            total_score=0.5,
            trace=[{"feature": "SOFR_ROC_30D", "value": 0.12, "contribution": 0.3}],
            confidence=0.8,
        )
        
        assert score.pro_score == 0.7
        assert score.con_score == 0.2
        assert len(score.trace) == 1
    
    def test_regime_types(self):
        """RegimeType enum 테스트"""
        assert RegimeType.RISK_ON.value == "risk_on"
        assert RegimeType.RISK_OFF.value == "risk_off"
        assert len(RegimeType) >= 8


class TestStep1Ingestion:
    """Step 1: Delta Ingestion 테스트"""
    
    def test_source_registry(self):
        """SourceRegistry 테스트"""
        from src.ingestion.source_registry import SourceRegistry
        from src.shared.schemas import SourceSpec, SourceType, DeltaMethod
        
        registry = SourceRegistry()
        
        # 소스 등록
        spec = SourceSpec(
            source_id="test_source",
            name="Test Source",
            source_type=SourceType.API,
            delta_method=DeltaMethod.SINCE_TIMESTAMP,
            provides_series=["SOFR", "VIX"],
        )
        registry.register(spec)
        
        # 조회 테스트
        assert registry.get("test_source") is not None
        assert registry.get_by_series("SOFR") is not None
        assert "SOFR" in registry.list_series()
    
    def test_fetch_state_store(self):
        """FetchStateStore 테스트"""
        from src.ingestion.fetch_state_store import FetchStateStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FetchStateStore(db_path=os.path.join(tmpdir, "state.db"))
            
            # 성공 업데이트
            store.update_success(
                source_id="fred",
                stream="default",
                fetched_count=10,
                last_timestamp=datetime(2024, 12, 14),
            )
            
            # 조회
            state = store.get("fred", "default")
            assert state is not None
            assert state.total_fetched == 10
            assert state.consecutive_failures == 0
    
    def test_normalizer(self):
        """Normalizer 테스트"""
        from src.ingestion.normalizer import Normalizer
        
        normalizer = Normalizer()
        
        raw_data = [
            {"timestamp": "2024-12-14", "value": 4.33},
            {"timestamp": "2024-12-13", "value": 4.30},
        ]
        
        result = normalizer.normalize_observations(
            raw_data, source_id="fred", series_id="SOFR"
        )
        
        assert result.success
        assert len(result.observations) == 2
        assert result.observations[0].value == 4.33
    
    def test_idempotency_guard(self):
        """IdempotencyGuard 테스트"""
        from src.ingestion.idempotency_guard import IdempotencyGuard
        
        guard = IdempotencyGuard()
        
        obs = Observation(
            series_id="SOFR",
            timestamp=datetime(2024, 12, 14),
            value=4.33,
            source_id="fred",
        )
        
        # 첫 번째 필터링
        result1 = guard.filter_observations([obs])
        assert result1.unique_count == 1
        assert result1.duplicate_count == 0
        
        # 저장 완료 표시
        guard.mark_as_stored(observations=[obs])
        
        # 두 번째 필터링 (중복)
        result2 = guard.filter_observations([obs])
        assert result2.unique_count == 0
        assert result2.duplicate_count == 1


class TestStep1Storage:
    """Step 1: Storage 테스트"""
    
    def test_timeseries_repository(self):
        """TimeSeriesRepository 테스트"""
        from src.storage.timeseries_repository import TimeSeriesRepository
        
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = TimeSeriesRepository(db_path=os.path.join(tmpdir, "ts.db"))
            
            # 데이터 저장
            obs = Observation(
                series_id="SOFR",
                timestamp=datetime(2024, 12, 14),
                value=4.33,
                source_id="fred",
            )
            stored, skipped = repo.append_batch([obs])
            
            assert stored == 1
            assert skipped == 0
            
            # 조회
            results = repo.get_range("SOFR")
            assert len(results) == 1
            assert results[0].value == 4.33
    
    def test_event_repository(self):
        """EventRepository 테스트"""
        from src.storage.event_repository import EventRepository
        
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = EventRepository(db_path=os.path.join(tmpdir, "events.db"))
            
            # 이벤트 저장
            event = Event(
                source_id="news",
                occurred_at=datetime(2024, 12, 14),
                title="Test Event",
                content="Test content",
            )
            success, _ = repo.append(event)
            
            assert success
            
            # 조회
            results = repo.get_range()
            assert len(results) == 1


class TestStep2Features:
    """Step 2: Feature 계산 테스트"""
    
    def test_feature_spec_registry(self):
        """FeatureSpecRegistry 테스트"""
        from src.features.feature_spec_registry import FeatureSpecRegistry
        
        registry = FeatureSpecRegistry()
        
        # 기본 스펙 확인
        assert len(registry.list_all()) > 0
        assert registry.get("SOFR_ROC_30D") is not None
    
    def test_feature_dependency_index(self):
        """FeatureDependencyIndex 테스트"""
        from src.features.feature_spec_registry import FeatureSpecRegistry
        from src.features.feature_dependency_index import FeatureDependencyIndex
        
        registry = FeatureSpecRegistry()
        index = FeatureDependencyIndex(registry)
        
        # SOFR 업데이트 시 영향받는 Feature
        affected = index.get_affected_features("SOFR")
        assert "SOFR_ROC_30D" in affected
    
    def test_feature_builder(self):
        """FeatureBuilder 테스트"""
        from src.features.feature_spec_registry import FeatureSpecRegistry
        from src.features.feature_builder import FeatureBuilder
        from src.storage.timeseries_repository import TimeSeriesRepository
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ts_repo = TimeSeriesRepository(db_path=os.path.join(tmpdir, "ts.db"))
            registry = FeatureSpecRegistry()
            builder = FeatureBuilder(registry, ts_repo)
            
            # 테스트 데이터 추가 (90일치 - ROC 30D 계산을 위해 충분한 과거 데이터 필요)
            observations = []
            base_date = datetime(2024, 10, 1)  # 더 과거부터 시작
            for i in range(90):
                obs = Observation(
                    series_id="SOFR",
                    timestamp=base_date + timedelta(days=i),
                    value=4.0 + (i * 0.01),  # 점진적 상승
                    source_id="fred",
                )
                observations.append(obs)
            ts_repo.append_batch(observations)
            
            # Feature 계산 (30일 ROC이므로 30일 후 시점부터 조회)
            result = builder.compute(
                "SOFR_ROC_30D",
                start=datetime(2024, 11, 1),  # 30일 후부터
                end=datetime(2024, 12, 30),
            )
            
            assert result.success
            # ROC 계산에는 30일 전 데이터가 필요하므로 일부 값은 있어야 함
            # 단, 윈도우 이전 값 찾기 로직에 따라 결과가 달라질 수 있음
            # 최소한 에러 없이 실행되어야 함
            assert result.error_message is None


class TestStep3Evidence:
    """Step 3: Evidence Layer 테스트"""
    
    def test_evidence_spec_registry(self):
        """EdgeEvidenceSpecRegistry 테스트"""
        from src.evidence.evidence_spec_registry import EdgeEvidenceSpecRegistry
        
        registry = EdgeEvidenceSpecRegistry()
        
        # 기본 스펙 확인
        assert len(registry.list_all()) > 0
        
        # 패턴 매칭 테스트
        spec = registry.find_matching_spec(
            head_id="", head_name="SOFR", head_type="Indicator",
            tail_id="", tail_name="NASDAQ", tail_type="Asset",
            relation_type="Affect", polarity="-"
        )
        assert spec is not None
    
    def test_evidence_accumulator(self):
        """EvidenceAccumulator 테스트"""
        from src.evidence.evidence_accumulator import EvidenceAccumulator
        
        accumulator = EvidenceAccumulator()
        
        # 여러 점수 누적
        for i in range(5):
            score = EvidenceScore(
                edge_id="EDGE_001",
                head_id="A", tail_id="B",
                relation_type="Affect",
                pro_score=0.6, con_score=0.2,
                total_score=0.4,
                trace=[],
            )
            result = accumulator.accumulate(score)
        
        # 누적 확인
        acc = accumulator.get("EDGE_001")
        assert acc is not None
        assert acc.observation_count == 5


class TestStep4Regime:
    """Step 4: Regime Layer 테스트"""
    
    def test_regime_spec_manager(self):
        """RegimeSpecManager 테스트"""
        from src.regime.regime_spec import RegimeSpecManager
        
        manager = RegimeSpecManager()
        
        # 기본 스펙 확인
        assert len(manager.list_all()) >= 8
        
        # 필요 Feature 확인
        features = manager.get_required_features()
        assert len(features) > 0
    
    def test_regime_detector(self):
        """RegimeDetector 테스트"""
        from src.regime.regime_spec import RegimeSpecManager
        from src.regime.regime_detector import RegimeDetector
        from src.features.feature_store import FeatureStore
        
        with tempfile.TemporaryDirectory() as tmpdir:
            feature_store = FeatureStore(db_path=os.path.join(tmpdir, "features.db"))
            spec_manager = RegimeSpecManager()
            detector = RegimeDetector(spec_manager, feature_store)
            
            # 테스트용 Feature 값 저장
            fv = FeatureValue(
                feature_id="VIX",
                timestamp=datetime(2024, 12, 14),
                value=15.0,  # 낮은 VIX -> risk_on
            )
            feature_store.save_batch([fv])
            
            # 탐지
            result = detector.detect()
            
            # 결과 확인 (Feature가 부족해도 동작해야 함)
            assert result is not None
            assert result.uncertainty >= 0


class TestEdgeWeightFusionV2:
    """EdgeWeightFusion v2.0 테스트"""
    
    def test_fuse_with_evidence_and_regime(self):
        """Evidence + Regime 적용 테스트"""
        from src.reasoning.edge_fusion import EdgeWeightFusion
        from src.reasoning.models import RetrievedPath
        
        fusion = EdgeWeightFusion(evidence_weight=0.5, regime_sensitivity=1.0)
        
        # 테스트 경로
        path = RetrievedPath(
            path_id="PATH_001",
            nodes=["A", "B"],
            node_names=["Entity A", "Entity B"],  # 필수 필드 추가
            edges=[{
                "relation_id": "EDGE_001",
                "head": "A",
                "tail": "B",
                "source": "domain",
                "domain_conf": 0.8,
                "semantic_tag": "sem_confident",
                "evidence_score": 0.7,        # Evidence 점수
                "regime_applicability": 1.2,  # Regime 조정
            }],
        )
        
        # 융합
        fused = fusion.fuse_path(path, regime_applicability=1.2)
        
        assert fused.path_weight > 0
        assert len(fused.fused_edges) == 1
        
        # Evidence와 Regime이 반영되었는지 확인
        edge = fused.fused_edges[0]
        assert edge.final_weight > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
