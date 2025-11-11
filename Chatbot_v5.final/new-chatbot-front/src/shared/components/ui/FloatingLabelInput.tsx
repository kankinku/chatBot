import React, { useState } from "react";

interface FloatingLabelInputProps {
  id: string;
  type?: string;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  label: string;
  required?: boolean;
  autoComplete?: string;
  readOnly?: boolean;
  icon?: React.ReactNode;
}

export const FloatingLabelInput: React.FC<FloatingLabelInputProps> = ({
  id,
  type = "text",
  value,
  onChange,
  label,
  required = false,
  autoComplete,
  readOnly = false,
  icon,
}) => {
  const [focused, setFocused] = useState(false);
  const isActive = focused || value.length > 0;

  return (
    <div className="relative">
      <input
        id={id}
        name={id}
        type={type}
        value={value}
        onChange={onChange}
        onFocus={() => setFocused(true)}
        onBlur={() => setFocused(false)}
        required={required}
        autoComplete={autoComplete}
        readOnly={readOnly}
        className={`peer w-full px-3 pt-6 pb-2 border rounded-md text-gray-900 placeholder-transparent focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200 ${
          focused ? "border-blue-500" : "border-gray-300"
        } ${readOnly ? "bg-gray-50" : "bg-white"} ${
          type === "date" && !value
            ? "[&::-webkit-datetime-edit-text]:opacity-0 [&::-webkit-datetime-edit-month-field]:opacity-0 [&::-webkit-datetime-edit-day-field]:opacity-0 [&::-webkit-datetime-edit-year-field]:opacity-0"
            : ""
        }`}
        placeholder={label}
      />
      <label
        htmlFor={id}
        className={`absolute left-3 transition-all duration-200 pointer-events-none ${
          isActive
            ? "top-1 text-xs text-blue-600"
            : "top-1/2 -translate-y-1/2 text-gray-500"
        }`}
      >
        {label}
      </label>
      {icon && (
        <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
          {icon}
        </div>
      )}
    </div>
  );
};
