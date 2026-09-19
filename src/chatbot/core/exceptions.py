"""
Exception Hierarchy

모든 예외를 계층 구조로 정의합니다.
눈가리고 아웅식 에러 처리를 방지합니다.
"""

from typing import Optional, Dict, Any


class ChatbotException(Exception):
    """
    챗봇 시스템의 베이스 예외
    
    모든 챗봇 관련 예외는 이 클래스를 상속받습니다.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        """
        Args:
            message: 에러 메시지
            error_code: 에러 코드 (예: "E001")
            details: 추가 정보
            cause: 원인이 된 예외
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "E000"
        self.details = details or {}
        self.cause = cause
    
    def to_dict(self) -> Dict[str, Any]:
        """예외를 딕셔너리로 변환 (로깅/API 응답용)"""
        result = {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
        }
        if self.details:
            result["details"] = self.details
        if self.cause:
            result["cause"] = str(self.cause)
        return result
    
    def __str__(self) -> str:
        parts = [f"[{self.error_code}] {self.message}"]
        if self.details:
            parts.append(f"Details: {self.details}")
        if self.cause:
            parts.append(f"Caused by: {self.cause}")
        return " | ".join(parts)


# ============================================================================
# Configuration Errors (E001-E099)
# ============================================================================

class ConfigurationError(ChatbotException):
    """설정 관련 에러"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E001",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


# ============================================================================
# Embedding Errors (E100-E199)
# ============================================================================

class EmbeddingError(ChatbotException):
    """임베딩 관련 에러"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E101",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class EmbeddingModelLoadError(EmbeddingError):
    """임베딩 모델 로드 실패"""
    
    def __init__(
        self,
        model_name: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"Failed to load embedding model: {model_name}",
            error_code="E101",
            details={"model_name": model_name},
            cause=cause,
        )


class EmbeddingGenerationError(EmbeddingError):
    """임베딩 생성 실패"""
    
    def __init__(
        self,
        text_sample: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            "Failed to generate embeddings",
            error_code="E102",
            details={"text_sample": text_sample[:100]},
            cause=cause,
        )


class EmbeddingDimensionMismatch(EmbeddingError):
    """임베딩 차원 불일치"""
    
    def __init__(
        self,
        expected: int,
        actual: int,
    ):
        super().__init__(
            f"Embedding dimension mismatch: expected {expected}, got {actual}",
            error_code="E103",
            details={"expected": expected, "actual": actual},
        )


# ============================================================================
# Retrieval Errors (E200-E299)
# ============================================================================

class RetrievalError(ChatbotException):
    """검색 관련 에러"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E201",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class VectorStoreNotFoundError(RetrievalError):
    """벡터 저장소를 찾을 수 없음"""
    
    def __init__(
        self,
        store_path: str,
    ):
        super().__init__(
            f"Vector store not found: {store_path}",
            error_code="E201",
            details={"store_path": store_path},
        )


class RetrievalTimeoutError(RetrievalError):
    """검색 타임아웃"""
    
    def __init__(
        self,
        query: str,
        timeout: int,
    ):
        super().__init__(
            f"Retrieval timeout after {timeout}s",
            error_code="E202",
            details={"query": query[:100], "timeout": timeout},
        )


class BM25IndexError(RetrievalError):
    """BM25 인덱스 에러"""
    
    def __init__(
        self,
        message: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"BM25 index error: {message}",
            error_code="E204",
            cause=cause,
        )


# ============================================================================
# Generation Errors (E300-E399)
# ============================================================================

class GenerationError(ChatbotException):
    """답변 생성 관련 에러"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E301",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class LLMConnectionError(GenerationError):
    """LLM 연결 실패"""
    
    def __init__(
        self,
        url: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"Failed to connect to LLM: {url}",
            error_code="E301",
            details={"url": url},
            cause=cause,
        )


class LLMTimeoutError(GenerationError):
    """LLM 타임아웃"""
    
    def __init__(
        self,
        timeout: int,
        prompt_length: int,
    ):
        super().__init__(
            f"LLM timeout after {timeout}s",
            error_code="E302",
            details={"timeout": timeout, "prompt_length": prompt_length},
        )


class LLMResponseEmptyError(GenerationError):
    """LLM 응답이 비어있음"""
    
    def __init__(
        self,
        prompt_sample: str,
    ):
        super().__init__(
            "LLM returned empty response",
            error_code="E303",
            details={"prompt_sample": prompt_sample[:100]},
        )


class LLMResponseInvalidError(GenerationError):
    """LLM 응답이 유효하지 않음"""
    
    def __init__(
        self,
        response_sample: str,
        reason: str,
    ):
        super().__init__(
            f"LLM response invalid: {reason}",
            error_code="E304",
            details={"response_sample": response_sample[:100], "reason": reason},
        )


# ============================================================================
# Preprocessing Errors (E400-E499)
# ============================================================================

class PreprocessingError(ChatbotException):
    """전처리 관련 에러"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E401",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class PDFLoadError(PreprocessingError):
    """PDF 로드 실패"""
    
    def __init__(
        self,
        file_path: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"Failed to load PDF: {file_path}",
            error_code="E401",
            details={"file_path": file_path},
            cause=cause,
        )


class TextExtractionError(PreprocessingError):
    """텍스트 추출 실패"""
    
    def __init__(
        self,
        file_path: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"Failed to extract text from: {file_path}",
            error_code="E402",
            details={"file_path": file_path},
            cause=cause,
        )


class OCRCorrectionError(PreprocessingError):
    """OCR 후처리 실패"""
    
    def __init__(
        self,
        text_sample: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            "OCR correction failed",
            error_code="E403",
            details={"text_sample": text_sample[:100]},
            cause=cause,
        )


# ============================================================================
# Chunking Errors (E500-E599)
# ============================================================================

class ChunkingError(ChatbotException):
    """청킹 관련 에러"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E501",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class InvalidChunkSizeError(ChunkingError):
    """잘못된 청크 크기"""
    
    def __init__(
        self,
        chunk_size: int,
        min_size: int,
        max_size: int,
    ):
        super().__init__(
            f"Invalid chunk size: {chunk_size} (must be between {min_size} and {max_size})",
            error_code="E502",
            details={"chunk_size": chunk_size, "min_size": min_size, "max_size": max_size},
        )


# ============================================================================
# Pipeline Errors (E600-E699)
# ============================================================================

class PipelineError(ChatbotException):
    """파이프라인 관련 에러"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E601",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class PipelineInitError(PipelineError):
    """파이프라인 초기화 실패"""
    
    def __init__(
        self,
        pipeline_name: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"Failed to initialize pipeline: {pipeline_name}",
            error_code="E601",
            details={"pipeline_name": pipeline_name},
            cause=cause,
        )


class PipelineExecutionError(PipelineError):
    """파이프라인 실행 실패"""
    
    def __init__(
        self,
        stage: str,
        cause: Optional[Exception] = None,
    ):
        super().__init__(
            f"Pipeline execution failed at stage: {stage}",
            error_code="E602",
            details={"stage": stage},
            cause=cause,
        )


# ============================================================================
# System Errors (E900-E999)
# ============================================================================

class SystemError(ChatbotException):
    """시스템 관련 에러"""
    
    def __init__(
        self,
        message: str,
        error_code: str = "E901",
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, error_code, details, cause)


class OutOfMemoryError(SystemError):
    """메모리 부족"""
    
    def __init__(
        self,
        operation: str,
    ):
        super().__init__(
            f"Out of memory during: {operation}",
            error_code="E901",
            details={"operation": operation},
        )


class DiskSpaceInsufficientError(SystemError):
    """디스크 공간 부족"""
    
    def __init__(
        self,
        required_mb: int,
        available_mb: int,
    ):
        super().__init__(
            f"Insufficient disk space: required {required_mb}MB, available {available_mb}MB",
            error_code="E902",
            details={"required_mb": required_mb, "available_mb": available_mb},
        )


class PermissionDeniedError(SystemError):
    """권한 거부"""
    
    def __init__(
        self,
        path: str,
        operation: str,
    ):
        super().__init__(
            f"Permission denied for {operation}: {path}",
            error_code="E903",
            details={"path": path, "operation": operation},
        )

