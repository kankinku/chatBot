import React, { useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";

interface LanguageSelectorProps {
  className?: string;
  showLabel?: boolean;
  onLanguageChange?: (languageCode: string) => void;
  autoSave?: boolean;
  selectedLanguage?: string;
}

interface LanguageOption {
  code: string;
  label: string;
  flag: string;
}

const LANGUAGE_OPTIONS: LanguageOption[] = [
  { code: "en", label: "English", flag: "🇺🇸" },
  { code: "ko", label: "한국어", flag: "🇰🇷" },
  { code: "es", label: "Español", flag: "🇪🇸" },
];

const LanguageSelector: React.FC<LanguageSelectorProps> = ({
  className = "",
  showLabel = true,
  onLanguageChange,
  autoSave = true,
  selectedLanguage,
}) => {
  const { i18n, t } = useTranslation();
  const [isOpen, setIsOpen] = useState(false);

  const handleLanguageChange = (languageCode: string) => {
    if (autoSave) {
      i18n.changeLanguage(languageCode);
    }

    if (onLanguageChange) {
      onLanguageChange(languageCode);
    }

    setIsOpen(false);
  };

  const currentLanguage = selectedLanguage || i18n.language || "en";
  const currentOption =
    LANGUAGE_OPTIONS.find((option) => option.code === currentLanguage) ||
    LANGUAGE_OPTIONS[0];

  return (
    <div className={`relative ${className}`}>
      {showLabel && (
        <label className="text-sm font-medium text-gray-600 mb-2 block">
          {t("profile.language")}
        </label>
      )}

      {/* Custom Dropdown Button */}
      <div className="relative">
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center justify-between w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm hover:border-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 min-w-[140px]"
        >
          <div className="flex items-center gap-2">
            <span className="text-lg">{currentOption.flag}</span>
            <span className="text-sm font-medium text-gray-700">
              {currentOption.label}
            </span>
          </div>
          <ChevronDown
            className={`w-4 h-4 text-gray-500 transition-transform duration-200 ${
              isOpen ? "transform rotate-180" : ""
            }`}
          />
        </button>

        {/* Dropdown Menu */}
        {isOpen && (
          <>
            {/* Backdrop */}
            <div
              className="fixed inset-0 z-10"
              onClick={() => setIsOpen(false)}
            />

            {/* Dropdown Content */}
            <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-20 overflow-hidden">
              {LANGUAGE_OPTIONS.map((option) => (
                <button
                  key={option.code}
                  type="button"
                  onClick={() => handleLanguageChange(option.code)}
                  className={`w-full px-3 py-2 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none transition-colors duration-150 flex items-center gap-2 ${
                    option.code === currentLanguage
                      ? "bg-blue-50 text-blue-700"
                      : "text-gray-700"
                  }`}
                >
                  <span className="text-lg">{option.flag}</span>
                  <span className="text-sm font-medium">{option.label}</span>
                  {option.code === currentLanguage && (
                    <span className="ml-auto text-blue-600">✓</span>
                  )}
                </button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default LanguageSelector;
