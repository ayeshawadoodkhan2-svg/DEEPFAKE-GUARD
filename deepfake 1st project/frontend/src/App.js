import React, { useState } from 'react';
import axios from 'axios';
import ImageUploader from './components/ImageUploader';
import Results from './components/Results';
import Header from './components/Header';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleImageUpload = (file) => {
    setImage(file);
    
    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
    
    // Clear previous results
    setResults(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!image) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', image);

      const response = await axios.post(
        `${API_BASE_URL}/predict`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      setResults(response.data);
    } catch (err) {
      console.error('Error:', err);
      setError(
        err.response?.data?.detail ||
        err.message ||
        'An error occurred during analysis'
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setImage(null);
    setImagePreview(null);
    setResults(null);
    setError(null);
  };

  return (
    <div className="App">
      <Header />
      <main className="main-content">
        <div className="container">
          {!results ? (
            <>
              <ImageUploader
                onImageUpload={handleImageUpload}
                imagePreview={imagePreview}
              />
              
              {error && (
                <div className="error-message">
                  <p>{error}</p>
                </div>
              )}
              
              <div className="button-group">
                <button
                  className="btn btn-primary"
                  onClick={handleAnalyze}
                  disabled={!imagePreview || loading}
                >
                  {loading ? 'Analyzing...' : 'Analyze Image'}
                </button>
              </div>
            </>
          ) : (
            <>
              <Results results={results} imagePreview={imagePreview} />
              <div className="button-group">
                <button className="btn btn-primary" onClick={handleReset}>
                  Analyze Another Image
                </button>
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
