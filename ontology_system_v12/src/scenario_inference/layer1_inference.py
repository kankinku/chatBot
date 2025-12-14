"""
Layer 1: Scenario Inference Core

추론 과정의 "결정 지점" 분해
각 모듈이 학습 가능한 결정을 내림
"""
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from src.scenario_inference.contracts import (
    ScenarioState,
    DecisionRecord,
    DecisionType,
    InferenceMode,
    EvidencePlan,
    EVIDENCE_PLAN_TEMPLATES,
    QueryFeatures,
    KGFeatures,
    EvidenceCoverage,
    TimeSeriesFeatures,
    BudgetFeatures,
    CandidateSummary,
    EvidenceTest,
)
from src.scenario_inference.layer0_guards import AsOfContextGuard, SnapshotResolver

logger = logging.getLogger(__name__)


# ============================================================
# ScenarioStateBuilder
# ============================================================

class ScenarioStateBuilder:
    """
    시나리오 상태 빌더
    
    책임:
    - 시나리오 추론에 필요한 내부 상태 구성 (피처/컨텍스트)
    
    실패모드:
    - 피처 드리프트 (버전 없음)
    - 미래 데이터 혼입 (치명)
    """
    
    def __init__(
        self,
        snapshot_resolver: Optional[SnapshotResolver] = None,
        as_of_guard: Optional[AsOfContextGuard] = None,
    ):
        self.snapshot_resolver = snapshot_resolver or SnapshotResolver()
        self.as_of_guard = as_of_guard or AsOfContextGuard()
    
    def build(
        self,
        query_text: str,
        as_of_time: datetime,
        query_id: Optional[str] = None,
        kg_store: Optional[Any] = None,
        evidence_store: Optional[Any] = None,
        ts_store: Optional[Any] = None,
        budget: Optional[BudgetFeatures] = None,
    ) -> ScenarioState:
        """
        시나리오 상태 구축
        
        Args:
            query_text: 질의 텍스트
            as_of_time: 기준 시점
            query_id: 질의 ID
            kg_store: KG 저장소
            evidence_store: 증거 저장소
            ts_store: 시계열 저장소
            budget: 비용 제약
        
        Returns:
            ScenarioState
        """
        # as_of 컨텍스트 설정
        AsOfContextGuard.set_context(as_of_time)
        
        # 스냅샷 참조 해석
        snapshots = self.snapshot_resolver.resolve_all(as_of_time)
        
        # Query 특성 추출
        query_features = self._extract_query_features(query_text)
        
        # KG 특성 추출
        kg_features = self._extract_kg_features(
            query_features.target_entities,
            kg_store,
            as_of_time,
        )
        
        # 증거 커버리지 추출
        evidence_coverage = self._extract_evidence_coverage(
            query_features.target_entities,
            evidence_store,
            as_of_time,
        )
        
        # 시계열 특성 추출
        ts_features = self._extract_ts_features(ts_store, as_of_time)
        
        state = ScenarioState(
            as_of_time=as_of_time,
            kg_snapshot_ref=snapshots["kg"].ref_id,
            ts_snapshot_ref=snapshots["timeseries"].ref_id,
            evidence_snapshot_ref=snapshots["evidence"].ref_id,
            query_id=query_id or "",
            query_text=query_text,
            query_features=query_features,
            kg_features=kg_features,
            evidence_coverage=evidence_coverage,
            ts_features=ts_features,
            budget=budget or BudgetFeatures(),
        )
        
        return state
    
    def _extract_query_features(self, query_text: str) -> QueryFeatures:
        """질의 특성 추출"""
        # 간단한 휴리스틱 추출
        entities = []
        entity_types = []
        
        # 키워드 기반 엔티티 추출 (실제로는 NER 사용)
        keywords = ["금리", "성장주", "VIX", "S&P500", "채권", "인플레이션"]
        for kw in keywords:
            if kw in query_text:
                entities.append(kw)
        
        # 질의 타입 추론
        query_type = "scenario"
        if "예측" in query_text or "앞으로" in query_text:
            query_type = "prediction"
        elif "왜" in query_text or "이유" in query_text:
            query_type = "analysis"
        elif "vs" in query_text or "비교" in query_text:
            query_type = "comparison"
        
        # 조건절 감지: "~면", "~경우", "~때", "if" 등
        import re
        condition_pattern = r'(으면|면|경우|때)(?:\s|$|[,.?!])'
        has_condition = bool(re.search(condition_pattern, query_text)) or "if" in query_text.lower()
        
        return QueryFeatures(
            query_type=query_type,
            target_entities=entities,
            entity_count=len(entities),
            entity_types=entity_types,
            has_condition=has_condition,
        )
    
    def _extract_kg_features(
        self,
        target_entities: List[str],
        kg_store: Optional[Any],
        as_of_time: datetime,
    ) -> KGFeatures:
        """KG 특성 추출"""
        # Stub - 실제로는 kg_store에서 조회
        return KGFeatures(
            target_entity_degrees={e: 5 for e in target_entities},
            path_candidate_count=len(target_entities) * 3,
            avg_path_length=2.5,
            domain_edge_ratio=0.8,
            personal_edge_ratio=0.2,
            conflict_edge_ratio=0.05,
            drift_flag_ratio=0.1,
        )
    
    def _extract_evidence_coverage(
        self,
        target_entities: List[str],
        evidence_store: Optional[Any],
        as_of_time: datetime,
    ) -> EvidenceCoverage:
        """증거 커버리지 추출"""
        # Stub
        return EvidenceCoverage(
            related_edge_count=len(target_entities) * 2,
            evidence_freshness_days=7.0,
            coverage_ratio=0.6,
            avg_evidence_strength=0.65,
            uncertainty_ratio=0.15,
        )
    
    def _extract_ts_features(
        self,
        ts_store: Optional[Any],
        as_of_time: datetime,
    ) -> TimeSeriesFeatures:
        """시계열 특성 추출"""
        # Stub
        return TimeSeriesFeatures(
            volatility_regime="medium",
            trend_signal="neutral",
            missing_ratio=0.05,
            regime_changes_30d=1,
        )


# ============================================================
# InferenceModeSelector
# ============================================================

@dataclass
class ModeDecision:
    """모드 선택 결과"""
    mode: InferenceMode
    rationale: str
    confidence: float
    estimated_cost: float


class InferenceModeSelector:
    """
    추론 모드 선택기
    
    책임:
    - 어떤 추론 모드를 쓸지 선택
    - (그래프만/그래프+수치증거/도메인만/도메인+퍼스널 보조 등)
    
    실패모드:
    - 비용 폭발 (항상 풀 모드)
    - 품질 저하 (항상 라이트 모드)
    """
    
    def __init__(
        self,
        policy_version: str = "rule_v1",
        learned_policy: Optional[Any] = None,
    ):
        self.policy_version = policy_version
        self.learned_policy = learned_policy
        
        # 모드별 예상 비용
        self.mode_costs = {
            InferenceMode.LIGHT: 1.0,
            InferenceMode.GRAPH_ONLY: 2.0,
            InferenceMode.DOMAIN_ONLY: 3.0,
            InferenceMode.GRAPH_EVIDENCE: 5.0,
            InferenceMode.DOMAIN_PERSONAL: 6.0,
            InferenceMode.FULL: 10.0,
        }
    
    def select(self, state: ScenarioState) -> Tuple[ModeDecision, DecisionRecord]:
        """
        모드 선택
        
        Returns:
            (ModeDecision, DecisionRecord)
        """
        candidates = self._generate_candidates(state)
        
        # 점수 계산
        scored_candidates = []
        for mode, rationale in candidates:
            score = self._score_mode(mode, state)
            scored_candidates.append((mode, rationale, score))
        
        # 정렬 및 선택
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        best_mode, best_rationale, best_score = scored_candidates[0]
        
        # 결정 기록 생성
        decision = ModeDecision(
            mode=best_mode,
            rationale=best_rationale,
            confidence=best_score,
            estimated_cost=self.mode_costs.get(best_mode, 5.0),
        )
        
        record = DecisionRecord(
            decision_type=DecisionType.MODE,
            candidates=[
                CandidateSummary(
                    candidate_id=m.value,
                    features={"cost": self.mode_costs.get(m, 5.0)},
                    score=s,
                )
                for m, r, s in scored_candidates
            ],
            candidate_count=len(scored_candidates),
            chosen_id=best_mode.value,
            scores={m.value: s for m, r, s in scored_candidates},
            selection_rationale=best_rationale,
            policy_version=self.policy_version,
            policy_type="rule" if not self.learned_policy else "learned",
            cost_estimate=decision.estimated_cost,
            confidence=best_score,
        )
        
        return decision, record
    
    def _generate_candidates(self, state: ScenarioState) -> List[Tuple[InferenceMode, str]]:
        """후보 모드 생성"""
        candidates = [
            (InferenceMode.LIGHT, "빠른 응답 필요"),
            (InferenceMode.GRAPH_ONLY, "그래프 구조 기반"),
            (InferenceMode.GRAPH_EVIDENCE, "그래프 + 수치증거"),
            (InferenceMode.DOMAIN_ONLY, "도메인 지식만"),
            (InferenceMode.DOMAIN_PERSONAL, "도메인 + 퍼스널"),
            (InferenceMode.FULL, "전체 모드"),
        ]
        return candidates
    
    def _score_mode(self, mode: InferenceMode, state: ScenarioState) -> float:
        """
        모드 점수 계산
        
        품질-비용 균형
        """
        cost = self.mode_costs.get(mode, 5.0)
        max_tests = state.budget.max_tests
        
        # 비용 제약 반영
        if cost > max_tests:
            return 0.1  # 비용 초과 시 낮은 점수
        
        # 상황별 점수
        base_score = 0.5
        
        # 증거 커버리지 높으면 EVIDENCE 모드 선호
        if mode in [InferenceMode.GRAPH_EVIDENCE, InferenceMode.FULL]:
            if state.evidence_coverage.coverage_ratio > 0.5:
                base_score += 0.2
        
        # 충돌 많으면 FULL 모드 선호
        if mode == InferenceMode.FULL:
            if state.kg_features.conflict_edge_ratio > 0.1:
                base_score += 0.15
        
        # 비용 효율성
        cost_efficiency = 1 - (cost / 10.0)
        
        return base_score + cost_efficiency * 0.2


# ============================================================
# PathCandidateGenerator
# ============================================================

@dataclass
class PathCandidate:
    """경로 후보"""
    path_id: str
    nodes: List[str]
    edges: List[Dict[str, Any]]
    path_length: int
    source: str  # domain, personal, mixed
    features: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0


class PathCandidateGenerator:
    """
    경로 후보 생성기
    
    책임:
    - BFS/제약으로 경로 후보 생성
    - 도메인 우선, 부족 시 퍼스널 보조
    
    실패모드:
    - 경로 폭발
    - 스푸리어스 경로 과다
    """
    
    def __init__(
        self,
        max_path_length: int = 4,
        max_candidates: int = 20,
        domain_priority: float = 0.8,
    ):
        self.max_path_length = max_path_length
        self.max_candidates = max_candidates
        self.domain_priority = domain_priority
    
    def generate(
        self,
        state: ScenarioState,
        mode: InferenceMode,
        kg_store: Optional[Any] = None,
    ) -> Tuple[List[PathCandidate], DecisionRecord]:
        """
        경로 후보 생성
        
        Returns:
            (PathCandidate 리스트, DecisionRecord)
        """
        all_candidates = []
        
        # 도메인 경로 생성
        domain_paths = self._generate_domain_paths(state, kg_store)
        all_candidates.extend(domain_paths)
        
        # 모드에 따라 퍼스널 경로 추가
        if mode in [InferenceMode.DOMAIN_PERSONAL, InferenceMode.FULL]:
            personal_paths = self._generate_personal_paths(state, kg_store)
            all_candidates.extend(personal_paths)
        
        # 피처라이징
        for candidate in all_candidates:
            candidate.features = self._featurize(candidate, state)
            candidate.score = self._score(candidate)
        
        # 정렬 및 상위 선택
        all_candidates.sort(key=lambda x: x.score, reverse=True)
        selected = all_candidates[:self.max_candidates]
        
        # 결정 기록
        record = DecisionRecord(
            decision_type=DecisionType.PATH_SELECT,
            candidates=[
                CandidateSummary(
                    candidate_id=c.path_id,
                    features=c.features,
                    score=c.score,
                )
                for c in all_candidates[:30]  # 상위 30개만 기록
            ],
            candidate_count=len(all_candidates),
            chosen_ids=[c.path_id for c in selected],
            scores={c.path_id: c.score for c in selected},
            selection_rationale=f"상위 {len(selected)}개 경로 선택",
            constraints_applied=["max_length", "spurious_filter"],
            confidence=selected[0].score if selected else 0.0,
        )
        
        return selected, record
    
    def _generate_domain_paths(
        self,
        state: ScenarioState,
        kg_store: Optional[Any],
    ) -> List[PathCandidate]:
        """도메인 경로 생성 (stub)"""
        # 실제로는 BFS로 KG 탐색
        return [
            PathCandidate(
                path_id=f"PATH_D_{i}",
                nodes=["A", "B", "C"][:2+i%2],
                edges=[{"relation": "Affect", "sign": "+"}],
                path_length=2 + i % 2,
                source="domain",
            )
            for i in range(5)
        ]
    
    def _generate_personal_paths(
        self,
        state: ScenarioState,
        kg_store: Optional[Any],
    ) -> List[PathCandidate]:
        """퍼스널 경로 생성 (stub)"""
        return [
            PathCandidate(
                path_id=f"PATH_P_{i}",
                nodes=["A", "X", "B"],
                edges=[{"relation": "Custom", "sign": "-"}],
                path_length=2,
                source="personal",
            )
            for i in range(2)
        ]
    
    def _featurize(self, candidate: PathCandidate, state: ScenarioState) -> Dict[str, float]:
        """경로 피처 추출"""
        return {
            "path_length": candidate.path_length,
            "is_domain": 1.0 if candidate.source == "domain" else 0.0,
            "evidence_coverage": 0.6 + (0.1 if candidate.source == "domain" else 0),
            "conflict_count": 0.0,
        }
    
    def _score(self, candidate: PathCandidate) -> float:
        """경로 점수 계산"""
        f = candidate.features
        
        # 도메인 우선
        domain_bonus = f.get("is_domain", 0) * self.domain_priority
        
        # 짧은 경로 선호
        length_penalty = f.get("path_length", 2) * 0.1
        
        # 증거 커버리지
        evidence_bonus = f.get("evidence_coverage", 0) * 0.3
        
        return 0.5 + domain_bonus + evidence_bonus - length_penalty


# ============================================================
# EvidencePlanGenerator
# ============================================================

class EvidencePlanGenerator:
    """
    증거 계획 생성기
    
    책임:
    - "검증 계획" 생성 (지표/윈도우/테스트/강건성 체크/비용 상한)
    
    실패모드:
    - 실행 불가 계획 (데이터 없음)
    - 과검증 (비용 폭발)
    - 테스트 편향
    """
    
    def __init__(self, policy_version: str = "rule_v1"):
        self.policy_version = policy_version
    
    def generate(
        self,
        paths: List[PathCandidate],
        state: ScenarioState,
        mode: InferenceMode,
    ) -> Tuple[EvidencePlan, DecisionRecord]:
        """
        증거 계획 생성
        
        Returns:
            (EvidencePlan, DecisionRecord)
        """
        # 모드에 따른 템플릿 선택
        template_candidates = self._select_template_candidates(mode, state)
        
        # 후보 평가
        scored_templates = []
        for template_name in template_candidates:
            template = EVIDENCE_PLAN_TEMPLATES.get(template_name)
            if template:
                score = self._score_template(template, state)
                scored_templates.append((template_name, template, score))
        
        scored_templates.sort(key=lambda x: x[2], reverse=True)
        best_name, best_template, best_score = scored_templates[0]
        
        # 선택된 템플릿 기반으로 구체적 테스트 생성
        plan = self._create_plan_from_template(best_template, paths, state)
        
        # 결정 기록
        record = DecisionRecord(
            decision_type=DecisionType.PLAN_SELECT,
            candidates=[
                CandidateSummary(
                    candidate_id=name,
                    features={"max_cost": t.max_cost, "robustness": 1.0 if t.include_robustness_check else 0.0},
                    score=s,
                )
                for name, t, s in scored_templates
            ],
            candidate_count=len(scored_templates),
            chosen_id=best_name,
            scores={name: s for name, t, s in scored_templates},
            selection_rationale=f"템플릿 '{best_name}' 선택 (점수: {best_score:.2f})",
            policy_version=self.policy_version,
            cost_estimate=plan.estimate_cost(),
            confidence=best_score,
        )
        
        return plan, record
    
    def _select_template_candidates(
        self,
        mode: InferenceMode,
        state: ScenarioState,
    ) -> List[str]:
        """모드에 따른 템플릿 후보 선택"""
        if mode == InferenceMode.LIGHT:
            return ["quick_check", "minimal"]
        elif mode in [InferenceMode.GRAPH_ONLY, InferenceMode.DOMAIN_ONLY]:
            return ["minimal", "standard"]
        elif mode in [InferenceMode.GRAPH_EVIDENCE, InferenceMode.DOMAIN_PERSONAL]:
            return ["standard", "comprehensive"]
        else:  # FULL
            return ["standard", "comprehensive", "robust"]
    
    def _score_template(self, template: EvidencePlan, state: ScenarioState) -> float:
        """템플릿 점수 계산"""
        # 비용 효율성
        cost_efficiency = 1 - (template.max_cost / 50.0)
        
        # 커버리지
        coverage_score = template.min_coverage_threshold
        
        # 강건성 필요 시
        robustness_bonus = 0.0
        if state.ts_features.volatility_regime == "high" and template.include_robustness_check:
            robustness_bonus = 0.2
        
        return cost_efficiency * 0.4 + coverage_score * 0.4 + robustness_bonus
    
    def _create_plan_from_template(
        self,
        template: EvidencePlan,
        paths: List[PathCandidate],
        state: ScenarioState,
    ) -> EvidencePlan:
        """템플릿 기반 구체적 계획 생성"""
        # model_copy 대신 명시적 생성으로 안정성 확보
        plan = EvidencePlan(
            plan_type=template.plan_type,
            max_cost=template.max_cost,
            max_time_seconds=template.max_time_seconds,
            include_robustness_check=template.include_robustness_check,
            alternative_window_sizes=template.alternative_window_sizes,
            early_stop_on_contradiction=template.early_stop_on_contradiction,
            min_coverage_threshold=template.min_coverage_threshold,
        )
        
        # 경로에서 엣지 추출하여 테스트 생성
        tests = []
        for path in paths[:5]:  # 상위 5개 경로
            for i, edge in enumerate(path.edges):
                try:
                    test = EvidenceTest(
                        target_edge_id=f"{path.path_id}_edge_{i}",
                        feature_id="ROC_30D",  # 기본값
                        test_type="direction",
                        window_days=30,
                        expected_direction=edge.get("sign", "+"),
                    )
                    tests.append(test)
                except Exception as e:
                    logger.warning(f"Failed to create test for edge: {e}")
                    continue
        
        plan.tests = tests[:int(template.max_cost)]  # 비용 제한
        plan.test_count = len(plan.tests)
        
        return plan


# ============================================================
# FusionScorer
# ============================================================

@dataclass
class ScoredScenario:
    """점수화된 시나리오"""
    scenario_id: str
    path: PathCandidate
    evidence_result: Dict[str, Any]
    scores: Dict[str, float]  # 각 차원별 점수
    final_score: float
    confidence: float


class FusionScorer:
    """
    융합 점수기
    
    책임:
    - EES(그래프) + PCS(퍼스널) + NES(수치검증) 합성
    - top-K 시나리오 채택
    
    실패모드:
    - 스칼라 한 방에 Goodhart
    - Domain/Personal 경계 붕괴
    """
    
    def __init__(
        self,
        policy_bundle: Optional[Dict[str, float]] = None,
        policy_version: str = "default_v1",
    ):
        self.policy_version = policy_version
        self.weights = policy_bundle or {
            "ees_domain": 0.35,
            "ees_personal": 0.15,
            "evidence_strength": 0.30,
            "evidence_robustness": 0.10,
            "path_length_penalty": 0.10,
        }
    
    def score(
        self,
        paths: List[PathCandidate],
        evidence_results: Dict[str, Any],
        state: ScenarioState,
        top_k: int = 3,
    ) -> Tuple[List[ScoredScenario], DecisionRecord]:
        """
        시나리오 점수화
        
        Returns:
            (ScoredScenario 리스트, DecisionRecord)
        """
        scored = []
        
        for path in paths:
            # 각 차원별 점수 계산
            scores = {
                "ees_domain": path.features.get("is_domain", 0) * 0.8 + 0.2,
                "ees_personal": (1 - path.features.get("is_domain", 0)) * 0.5,
                "evidence_strength": evidence_results.get(path.path_id, {}).get("strength", 0.5),
                "evidence_robustness": evidence_results.get(path.path_id, {}).get("robustness", 0.5),
                "path_length_penalty": max(0, 1 - path.path_length * 0.15),
            }
            
            # 가중합
            final_score = sum(
                self.weights.get(k, 0) * v
                for k, v in scores.items()
            )
            
            # 신뢰도 계산
            confidence = min(1.0, final_score * 1.2)
            
            scored.append(ScoredScenario(
                scenario_id=f"SCEN_{path.path_id}",
                path=path,
                evidence_result=evidence_results.get(path.path_id, {}),
                scores=scores,
                final_score=final_score,
                confidence=confidence,
            ))
        
        # 정렬 및 top-K
        scored.sort(key=lambda x: x.final_score, reverse=True)
        selected = scored[:top_k]
        
        # 결정 기록
        record = DecisionRecord(
            decision_type=DecisionType.FUSION_SCORE,
            candidates=[
                CandidateSummary(
                    candidate_id=s.scenario_id,
                    features=s.scores,
                    score=s.final_score,
                )
                for s in scored[:10]
            ],
            candidate_count=len(scored),
            chosen_ids=[s.scenario_id for s in selected],
            scores={s.scenario_id: s.final_score for s in scored},
            selection_rationale=f"Top-{top_k} 선택",
            policy_version=self.policy_version,
            confidence=selected[0].confidence if selected else 0.0,
            metadata={"weights": self.weights},
        )
        
        return selected, record
