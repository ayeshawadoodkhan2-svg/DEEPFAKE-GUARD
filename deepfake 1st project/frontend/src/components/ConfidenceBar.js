import React from 'react';

function ConfidenceBar({ confidence }) {
  const getColor = (conf) => {
    if (conf >= 75) return '#ef4444';  // Red for high confidence
    if (conf >= 50) return '#eab308';  // Yellow for medium-high
    if (conf >= 25) return '#3b82f6';  // Blue for medium
    return '#10b981';  // Green for low
  };

  return (
    <div className="confidence-bar-container">
      <div className="confidence-bar-background">
        <div
          className="confidence-bar-fill"
          style={{
            width: `${confidence}%`,
            backgroundColor: getColor(confidence),
          }}
        ></div>
      </div>
    </div>
  );
}

export default ConfidenceBar;
