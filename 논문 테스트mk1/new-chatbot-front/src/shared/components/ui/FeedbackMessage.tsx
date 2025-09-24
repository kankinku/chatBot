import React, { useEffect, useRef } from 'react';
import { CheckCircle, AlertCircle, Info, AlertTriangle, X } from 'lucide-react';

interface FeedbackMessageProps {
  type: 'success' | 'error' | 'info' | 'warning' | null;
  message: string;
  isVisible: boolean;
  onDismiss?: () => void;
  className?: string;
  showIcon?: boolean;
  showCloseButton?: boolean;
  autoHideDuration?: number;
  role?: string;
}

/**
 * Component for displaying feedback messages to the user
 * 
 * Features:
 * - Multiple message types (success, error, info, warning)
 * - Auto-hide functionality with configurable duration
 * - Customizable appearance with icons and close button
 * - Accessibility support with proper ARIA attributes
 */
export const FeedbackMessage: React.FC<FeedbackMessageProps> = ({
  type,
  message,
  isVisible,
  onDismiss,
  className = '',
  showIcon = true,
  showCloseButton = true,
  autoHideDuration,
  role = 'alert',
}) => {
  // Reference to the auto-hide timer
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  
  // Auto-hide effect with cleanup
  useEffect(() => {
    // Clear any existing timer when props change
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    
    // Set up new timer if conditions are met
    if (isVisible && autoHideDuration && onDismiss) {
      timerRef.current = setTimeout(() => {
        onDismiss();
        timerRef.current = null;
      }, autoHideDuration);
    }
    
    // Cleanup on unmount or when dependencies change
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [isVisible, autoHideDuration, onDismiss, message, type]);

  // Don't render anything if not visible or no type
  if (!isVisible || !type) return null;

  // Determine styles based on type
  const styles = {
    success: {
      bg: 'bg-green-50',
      border: 'border-green-200',
      text: 'text-green-700',
      icon: <CheckCircle size={16} className="text-green-500" />,
    },
    error: {
      bg: 'bg-red-50',
      border: 'border-red-200',
      text: 'text-red-700',
      icon: <AlertCircle size={16} className="text-red-500" />,
    },
    info: {
      bg: 'bg-blue-50',
      border: 'border-blue-200',
      text: 'text-blue-700',
      icon: <Info size={16} className="text-blue-500" />,
    },
    warning: {
      bg: 'bg-yellow-50',
      border: 'border-yellow-200',
      text: 'text-yellow-700',
      icon: <AlertTriangle size={16} className="text-yellow-500" />,
    },
  }[type];

  return (
    <div
      className={`flex items-center justify-between p-3 rounded-lg border ${styles.bg} ${styles.border} ${styles.text} ${className}`}
      role={role}
      aria-live="polite"
    >
      <div className="flex items-center space-x-2">
        {showIcon && styles.icon}
        <span className="text-sm">{message}</span>
      </div>
      
      {showCloseButton && onDismiss && (
        <button
          onClick={onDismiss}
          className="ml-auto text-gray-400 hover:text-gray-600 focus:outline-none"
          aria-label="Close"
          type="button"
        >
          <X size={16} />
        </button>
      )}
    </div>
  );
};