// App.jsx
// Main React component for the Jasmine Flower Detection and Counting System.

import { useState, useRef } from 'react'

function App() {
  const [selectedFile, setSelectedFile]   = useState(null)
  const [previewUrl, setPreviewUrl]       = useState(null)
  const [confidence, setConfidence]       = useState(0.25)
  const [loading, setLoading]             = useState(false)
  const [result, setResult]               = useState(null)
  const [error, setError]                 = useState(null)
  const [activeTab, setActiveTab]         = useState('comparison')
  const fileInputRef                      = useRef(null)

  // ── Handle file selection ──────────────────────────────────────────────────
  const handleFileChange = (file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (JPG, PNG, WEBP)')
      return
    }
    setError(null)
    setResult(null)
    setSelectedFile(file)
    setPreviewUrl(URL.createObjectURL(file))
  }

  const handleDrop = (e) => {
    e.preventDefault()
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0])
    }
  }

  const handleDragOver = (e) => e.preventDefault()

  // ── Send image to Flask API ────────────────────────────────────────────────
  const handleDetect = async () => {
    if (!selectedFile) {
      setError('Please upload an image first.')
      return
    }
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('image', selectedFile)
    formData.append('confidence', confidence)

    try {
      const response = await fetch('/detect', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Detection failed.')
      }
      setResult(data)
      setActiveTab('comparison')
    } catch (err) {
      setError(err.message || 'An unexpected error occurred.')
    } finally {
      setLoading(false)
    }
  }

  // ── Reset everything ───────────────────────────────────────────────────────
  const handleReset = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="app-container">

      {/* Header */}
      <header className="navbar">
        <div className="nav-brand">
          <span className="brand-icon">🌿</span>
          <div>
            <h1>Jasmine Flower Detection & Counting System</h1>
          </div>
        </div>
        <div className="status-badge">
          <span className="status-dot"></span>
          <span>Model Ready</span>
        </div>
      </header>

      <main className="main-content">

        {/* ── Left Column: Upload & Controls ── */}
        <div className="control-card">
          <h2 className="section-title">Upload Image</h2>
          <p className="section-desc">
            Upload any flower image. The system detects and counts{' '}
            <strong>Jasmine flowers</strong>
          </p>

          {/* Drag and Drop Zone */}
          <div
            className={`dropzone ${previewUrl ? 'has-preview' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onClick={() => fileInputRef.current && fileInputRef.current.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: 'none' }}
              accept="image/jpeg,image/png,image/webp,image/jpg"
              onChange={(e) => handleFileChange(e.target.files[0])}
            />
            {previewUrl ? (
              <div className="preview-box">
                <img src={previewUrl} alt="Preview" className="preview-img" />
                <div className="preview-overlay">
                  <span>Click or drag to change image</span>
                </div>
              </div>
            ) : (
              <div className="dropzone-prompt">
                <div className="upload-icon">🌸</div>
                <h3>Drag & Drop your image here</h3>
                <p>or click to browse from your computer</p>
                <span className="file-hints">Supports JPG, PNG, WEBP (up to 16MB)</span>
              </div>
            )}
          </div>

          {/* Confidence Slider */}
          <div className="slider-control">
            <div className="slider-label">
              <span>Detection Confidence</span>
              <strong>{Math.round(confidence * 100)}%</strong>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={confidence}
              onChange={(e) => setConfidence(parseFloat(e.target.value))}
            />
            <div className="slider-hints">
              <span>More sensitive</span>
              <span>Stricter</span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="action-row">
            <button
              className="btn btn-primary"
              onClick={handleDetect}
              disabled={!selectedFile || loading}
            >
              {loading ? (
                <><span className="spinner"></span> Detecting Jasmine...</>
              ) : (
                <>🔍 Detect Jasmine</>
              )}
            </button>
            {(selectedFile || result) && (
              <button
                className="btn btn-secondary"
                onClick={handleReset}
                disabled={loading}
              >
                Reset
              </button>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="alert alert-error">
              <strong>Notice:</strong> {error}
            </div>
          )}
        </div>

        {/* ── Right Column: Results ── */}
        <div className="results-card">
          <h2 className="section-title">Detection Results</h2>

          {/* Loading state */}
          {loading && (
            <div className="loading-state">
              <div className="floral-spinner"></div>
              <h3>Analyzing image with YOLO...</h3>
              <p>Identifying jasmine features and counting flowers</p>
            </div>
          )}

          {/* Empty state */}
          {!loading && !result && (
            <div className="empty-state">
              <div className="empty-icon">🌼</div>
              <h3>No results yet</h3>
              <p>Upload a flower photo and click Detect Jasmine.</p>
            </div>
          )}

          {/* Results */}
          {!loading && result && (
            <div className="results-display">

              {/* Stats Banner */}
              <div className="stats-banner">
                <div className="stat-box primary">
                  <span className="stat-label">Total Jasmine Flowers Detected</span>
                  <div className="stat-value-row">
                    <span className="stat-count">{result.jasmine_count}</span>
                    <span className="stat-badge">
                      {result.jasmine_count === 1 ? '1 Flower' : `${result.jasmine_count} Flowers`}
                    </span>
                  </div>
                </div>
                <div className="stat-box secondary">
                  <span className="stat-label">Filter Status</span>
                  <span className="stat-subtext">Class: <strong>Jasmine Only</strong></span>
                  <span className="stat-subtext">Other Flowers: <strong>Excluded</strong></span>
                </div>
              </div>

              {/* View Tabs */}
              <div className="view-tabs">
                {['comparison', 'annotated', 'original'].map((tab) => (
                  <button
                    key={tab}
                    className={`tab-btn ${activeTab === tab ? 'active' : ''}`}
                    onClick={() => setActiveTab(tab)}
                  >
                    {tab === 'comparison' ? 'Side-by-Side'
                      : tab === 'annotated' ? 'Detected Image'
                      : 'Original Image'}
                  </button>
                ))}
              </div>

              {/* Image Display */}
              <div className="image-view-area">
                {activeTab === 'comparison' && (
                  <div className="comparison-grid">
                    <div className="img-card">
                      <h4>Original Upload</h4>
                      <img src={result.original_image_url} alt="Original" />
                    </div>
                    <div className="img-card">
                      <h4>Detected Jasmine</h4>
                      <img src={result.processed_image_url} alt="Annotated" />
                    </div>
                  </div>
                )}
                {activeTab === 'annotated' && (
                  <div className="single-img-card">
                    <img src={result.processed_image_url} alt="Annotated" />
                  </div>
                )}
                {activeTab === 'original' && (
                  <div className="single-img-card">
                    <img src={result.original_image_url} alt="Original" />
                  </div>
                )}
              </div>

              {/* Detection Details */}
              {result.detections && result.detections.length > 0 && (
                <div className="detection-details">
                  <h4>Detection Details ({result.detections.length} boxes)</h4>
                  <div className="detection-chips">
                    {result.detections.map((d, idx) => (
                      <span key={idx} className="chip">
                        Flower #{idx + 1}: {(d.confidence * 100).toFixed(1)}% conf
                      </span>
                    ))}
                  </div>
                </div>
              )}

            </div>
          )}
        </div>

      </main>

      <footer className="footer">
        <p>Jasmine Flower Detection and Counting System </p>
      </footer>

    </div>
  )
}

export default App
