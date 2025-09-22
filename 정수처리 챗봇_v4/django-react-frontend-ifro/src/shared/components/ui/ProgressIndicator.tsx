import React from 'react';

interface ProgressIndicatorProps {
  progress: number;
  size?: 'sm' | 'md' | 'lg';
  showPercentage?: boolean;
  color?: string;
  trackColor?: string;
  className?: string;
}

export const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  progress,
  size = 'md',
  showPercentage = true,
  color = 'bg-blue-600',
  trackColor = 'bg-gray-200',
  className = '',
}) => {
  // Determine height based on size
  const heightClass = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  }[size];

  // Ensure progress is between 0 and 100, handle NaN
  const normalizedProgress = isNaN(progress) ? 0 : Math.min(100, Math.max(0, progress));
  
  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className={`flex-grow ${trackColor} rounded-full ${heightClass}`}>
        <div
          className={`${color} ${heightClass} rounded-full transition-all duration-300 ease-in-out`}
          style={{ width: `${normalizedProgress}%` }}
          role="progressbar"
          aria-valuenow={normalizedProgress}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
      {showPercentage && (
        <span className="text-sm text-gray-600 font-medium min-w-[40px] text-right">
          {Math.round(normalizedProgress)}%
        </span>
      )}
    </div>
  );
};