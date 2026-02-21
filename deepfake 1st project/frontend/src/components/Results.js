import React from 'react';
import ConfidenceBar from './ConfidenceBar';

function Results({ results, imagePreview }) {
  const isDeepfake = results.prediction === 'Deepfake';
  const confidence = results.confidence;

  return (
    <div className="results-section">
      <div className="results-container">
        <div className="results-image">
          <img src={imagePreview} alt="Analyzed" className="result-image" />
          <div className={`badge ${isDeepfake ? 'badge-deepfake' : 'badge-real'}`}>
            {results.prediction}
          </div>
        </div>

        <div className="results-content">
          <div className="result-card">
            <h2>Analysis Results</h2>
            
            <div className="result-item">
              <label>Prediction:</label>
              <p className={`result-value ${isDeepfake ? 'text-danger' : 'text-success'}`}>
                {results.prediction}
              </p>
            </div>

            <div className="result-item">
              <label>Confidence Score:</label>
              <ConfidenceBar confidence={confidence} />
              <p className="confidence-text">{confidence.toFixed(2)}%</p>
            </div>

            <div className="result-item">
              <label>Analysis:</label>
              <p>{results.explanation}</p>
            </div>

            <div className="result-item">
              <label>Model:</label>
              <p>{results.model}</p>
            </div>

            <div className="result-item">
              <label>Processed:</label>
              <p>{results.filename}</p>
            </div>

            {results.grad_cam_available && (
              <div className="result-item">
                <label>Visualization:</label>
                <p className="success-text">✓ Grad-CAM heatmap generated</p>
              </div>
            )}
          </div>

          <div className="explanation-box">
            <h3>What does this mean?</h3>
            {isDeepfake ? (
              <p>
                This image has been classified as a <strong>Deepfake</strong> with 
                <strong> {confidence.toFixed(2)}%</strong> confidence. The AI model 
                detected signs of manipulation or synthetic generation. This could include 
                face swaps, expression reenactment, or other facial manipulations.
              </p>
            ) : (
              <p>
                This image has been classified as <strong>Real</strong> with 
                <strong> {confidence.toFixed(2)}%</strong> confidence. The AI model found 
                no significant signs of manipulation. However, always verify critical 
                images through multiple sources.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Results;
