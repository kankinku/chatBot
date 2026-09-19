import React, { useState } from 'react';
import { processPDFs } from '../../../shared/services/chat';
import './ProcessPDFButton.css';

export const ProcessPDFButton: React.FC = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [result, setResult] = useState<{
    processed_files: string[];
    skipped_files: string[];
    total_chunks: number;
    processing_time_seconds: number;
  } | null>(null);
  const [showResult, setShowResult] = useState(true);

  const handleProcess = async () => {
    setIsProcessing(true);
    setProgress(0);
    setMessage('PDF 처리 시작...');
    setResult(null);
    setShowResult(true);

    try {
      const response = await processPDFs((progressValue, progressMessage) => {
        setProgress(progressValue);
        setMessage(progressMessage);
      });
      setProgress(100);
      setResult({
        processed_files: response.processed_files,
        skipped_files: response.skipped_files,
        total_chunks: response.total_chunks,
        processing_time_seconds: response.processing_time_seconds,
      });
      setMessage(response.message);
    } catch (error: any) {
      setMessage(`오류: ${error.message}`);
      setResult(null);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="process-pdf-container">
      <button
        className="process-pdf-button"
        onClick={handleProcess}
        disabled={isProcessing}
      >
        {isProcessing ? '처리 중...' : 'PDF 처리'}
      </button>

      {isProcessing && (
        <div className="progress-bar-container">
          <div className="progress-bar">
            <div 
              className="progress-bar-fill" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <div className="progress-text">{progress}%</div>
        </div>
      )}

      {message && (
        <div className={`process-message ${message.includes('오류') ? 'error' : isProcessing ? 'info' : 'success'}`}>
          {message}
        </div>
      )}

      {result && showResult && (
        <div 
          className="process-result"
          onClick={() => setShowResult(false)}
          style={{ cursor: 'pointer' }}
          title="클릭하여 닫기"
        >
          {result.processed_files.length > 0 && (
            <div className="result-section">
              <strong>처리된 파일 ({result.processed_files.length}개):</strong>
              <ul>
                {result.processed_files.map((file, idx) => (
                  <li key={idx}>{file}</li>
                ))}
              </ul>
            </div>
          )}
          
          {result.skipped_files.length > 0 && (
            <div className="result-section">
              <strong>스킵된 파일 ({result.skipped_files.length}개):</strong>
              <ul>
                {result.skipped_files.map((file, idx) => (
                  <li key={idx}>{file}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="result-stats">
            <div>전체 청크 수: {result.total_chunks}개</div>
            <div>처리 시간: {result.processing_time_seconds.toFixed(2)}초</div>
          </div>
          
          <div className="result-close-hint">💡 클릭하여 닫기</div>
        </div>
      )}
    </div>
  );
};

