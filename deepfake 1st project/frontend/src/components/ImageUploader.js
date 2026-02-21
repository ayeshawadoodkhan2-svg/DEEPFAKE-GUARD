import React, { useState } from 'react';

function ImageUploader({ onImageUpload, imagePreview }) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      onImageUpload(files[0]);
    }
  };

  const handleFileSelect = (e) => {
    const files = e.target.files;
    if (files.length > 0) {
      onImageUpload(files[0]);
    }
  };

  return (
    <div className="uploader-section">
      {!imagePreview ? (
        <div
          className={`dropzone ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="dropzone-content">
            <div className="upload-icon">📤</div>
            <h2>Drag and drop your image</h2>
            <p>or click below to select</p>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="file-input"
              id="file-input"
            />
            <label htmlFor="file-input" className="btn btn-secondary">
              Choose Image
            </label>
          </div>
        </div>
      ) : (
        <div className="preview-section">
          <h2>Image Preview</h2>
          <img src={imagePreview} alt="Preview" className="preview-image" />
          <input
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="file-input"
            id="file-input-new"
          />
          <label htmlFor="file-input-new" className="btn btn-secondary">
            Change Image
          </label>
        </div>
      )}
    </div>
  );
}

export default ImageUploader;
