import React, { useState } from 'react';
import styled from 'styled-components';
import { DateRangeSelection } from '../types';
import { getTomorrowDate, getDateFromToday, validateDateRange } from '../api/csvDataService';

const DateRangeContainer = styled.div`
  display: flex;
  flex-direction: column;
  gap: 15px;
  padding: 20px;
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-bottom: 20px;
`;

const DateRangeHeader = styled.h3`
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
  margin: 0;
`;

const DateInputRow = styled.div`
  display: flex;
  gap: 15px;
  align-items: center;
  flex-wrap: wrap;
`;

const DateInputGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 5px;
`;

const DateLabel = styled.label`
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const DateInput = styled.input`
  padding: 8px 12px;
  background: var(--background);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 14px;
  min-width: 140px;
  
  &:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(247, 147, 26, 0.1);
  }
  
  &::-webkit-calendar-picker-indicator {
    filter: invert(1);
    cursor: pointer;
  }
`;

const ApplyButton = styled.button`
  padding: 8px 16px;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover:not(:disabled) {
    background: var(--accent-hover);
    transform: translateY(-1px);
  }
  
  &:disabled {
    background: var(--text-secondary);
    cursor: not-allowed;
    opacity: 0.6;
  }
`;

const ErrorMessage = styled.div`
  color: #ff6b6b;
  font-size: 12px;
  margin-top: 5px;
`;

interface DateRangeSelectorProps {
  onDateRangeChange: (dateRange: DateRangeSelection) => void;
  disabled?: boolean;
}

const DateRangeSelector: React.FC<DateRangeSelectorProps> = ({ 
  onDateRangeChange, 
  disabled = false 
}) => {
  const [startDate, setStartDate] = useState(getTomorrowDate());
  const [endDate, setEndDate] = useState(getDateFromToday(30));
  const [error, setError] = useState<string | null>(null);

  const handleApply = () => {
    const validationError = validateDateRange(startDate, endDate);
    
    if (validationError) {
      setError(validationError);
      return;
    }
    
    setError(null);
    onDateRangeChange({ startDate, endDate });
  };

  const minDate = getTomorrowDate();
  const maxDate = getDateFromToday(1095);

  return (
    <DateRangeContainer>
      <DateRangeHeader>Custom Date Range</DateRangeHeader>
      
      <DateInputRow>
        <DateInputGroup>
          <DateLabel>From Date</DateLabel>
          <DateInput
            type="date"
            value={startDate}
            min={minDate}
            max={maxDate}
            onChange={(e) => {
              setStartDate(e.target.value);
              setError(null);
            }}
            disabled={disabled}
          />
        </DateInputGroup>
        
        <DateInputGroup>
          <DateLabel>To Date</DateLabel>
          <DateInput
            type="date"
            value={endDate}
            min={minDate}
            max={maxDate}
            onChange={(e) => {
              setEndDate(e.target.value);
              setError(null);
            }}
            disabled={disabled}
          />
        </DateInputGroup>
        
        <ApplyButton 
          onClick={handleApply} 
          disabled={disabled || !startDate || !endDate}
        >
          Apply Range
        </ApplyButton>
      </DateInputRow>
      
      {error && <ErrorMessage>{error}</ErrorMessage>}
    </DateRangeContainer>
  );
};

export default DateRangeSelector;
