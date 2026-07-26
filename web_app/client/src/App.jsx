import React, { useState, useRef, useEffect } from 'react';
import './index.css';

const API_BASE = 'http://localhost:5001/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('canvas'); // 'canvas' | 'upload' | 'synthetic'
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  // Canvas State
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushWidth, setBrushWidth] = useState(6);

  // Upload State
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  // Synthetic Text State
  const [synthText, setSynthText] = useState('বাংলাদেশ');

  // Fetch Health Check Status
  useEffect(() => {
    fetchHealth();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealth(data);
    } catch (err) {
      console.warn("API Server connection warning:", err);
      setHealth({ status: "connecting", activeCheckpoint: "Checking Server..." });
    }
  };

  // ----------------------------------------------------
  // Canvas Handlers
  // ----------------------------------------------------
  useEffect(() => {
    if (activeTab === 'canvas' && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      // Fill white background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }, [activeTab]);

  const startDrawing = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches?.[0]?.clientX) - rect.left;
    const y = (e.clientY || e.touches?.[0]?.clientY) - rect.top;

    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = brushWidth;
    setIsDrawing(true);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.touches?.[0]?.clientX) - rect.left;
    const y = (e.clientY || e.touches?.[0]?.clientY) - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
    setError(null);
  };

  const predictCanvas = async () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const imageBase64 = canvas.toDataURL('image/png');

    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/predict/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageBase64 })
      });
      const data = await res.json();

      if (data.success) {
        setPrediction(data.prediction);
      } else {
        setError(data.error || "Prediction failed.");
      }
    } catch (err) {
      setError("Failed to connect to HTR API Server on localhost:5001");
    } finally {
      setLoading(false);
    }
  };

  // ----------------------------------------------------
  // File Upload Handlers
  // ----------------------------------------------------
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError(null);
    }
  };

  const predictImageFile = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const res = await fetch(`${API_BASE}/predict/image`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.success) {
        setPrediction(data.prediction);
      } else {
        setError(data.error || "Prediction failed.");
      }
    } catch (err) {
      setError("Failed to connect to HTR API Server on localhost:5001");
    } finally {
      setLoading(false);
    }
  };

  // ----------------------------------------------------
  // Synthetic Text Handlers
  // ----------------------------------------------------
  const predictSyntheticText = async () => {
    if (!synthText.trim()) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/predict/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: synthText })
      });
      const data = await res.json();

      if (data.success) {
        setPrediction(data.prediction);
      } else {
        setError(data.error || "Prediction failed.");
      }
    } catch (err) {
      setError("Failed to connect to HTR API Server on localhost:5001");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* App Header */}
      <header className="app-header">
        <div className="badge-sota">⚡ SOTA Hybrid Model • ConvNeXt + BiLSTM + RL Agent</div>
        <h1>Bangla <span>Handwritten Text Recognition</span></h1>
        <p>Offline Bangla OCR with Actor-Critic Reinforcement Learning & Trie Lexicon Post-Processing</p>
      </header>

      {/* Model Health Status Banner */}
      <div className="status-banner">
        <div className="status-info">
          <div className="status-dot"></div>
          <span>Model Status: <strong>{health?.activeCheckpoint || "Online"}</strong></span>
        </div>
        <div className="status-chip">Apple Silicon Acceleration (MPS Enabled)</div>
      </div>

      {/* Main Grid */}
      <div className="main-grid">
        {/* Input Card */}
        <div className="card">
          <div className="card-title">✏️ Input Mode</div>

          {/* Tabs */}
          <div className="tabs-header">
            <button
              className={`tab-btn ${activeTab === 'canvas' ? 'active' : ''}`}
              onClick={() => setActiveTab('canvas')}
            >
              Draw Canvas
            </button>
            <button
              className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              Upload Image
            </button>
            <button
              className={`tab-btn ${activeTab === 'synthetic' ? 'active' : ''}`}
              onClick={() => setActiveTab('synthetic')}
            >
              Test Text
            </button>
          </div>

          {/* Tab 1: Interactive Canvas */}
          {activeTab === 'canvas' && (
            <div className="canvas-wrapper">
              <div className="canvas-container">
                <canvas
                  ref={canvasRef}
                  width={500}
                  height={200}
                  onMouseDown={startDrawing}
                  onMouseMove={draw}
                  onMouseUp={stopDrawing}
                  onMouseLeave={stopDrawing}
                  onTouchStart={startDrawing}
                  onTouchMove={draw}
                  onTouchEnd={stopDrawing}
                />
              </div>
              <div className="canvas-tools">
                <div className="brush-slider">
                  <span>Stroke Thickness:</span>
                  <input
                    type="range"
                    min="2"
                    max="14"
                    value={brushWidth}
                    onChange={(e) => setBrushWidth(Number(e.target.value))}
                  />
                  <span>{brushWidth}px</span>
                </div>
                <button className="btn-secondary" onClick={clearCanvas}>Clear Canvas</button>
              </div>
              <button className="btn-primary" onClick={predictCanvas} disabled={loading}>
                {loading ? <div className="spinner"></div> : "Recognize Handwriting"}
              </button>
            </div>
          )}

          {/* Tab 2: Upload Image File */}
          {activeTab === 'upload' && (
            <div className="canvas-wrapper">
              <label className="dropzone" style={{ width: '100%' }}>
                <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
                {previewUrl ? (
                  <img src={previewUrl} alt="Handwriting Preview" className="preview-image" />
                ) : (
                  <div>
                    <div style={{ fontSize: '2rem', marginBottom: '8px' }}>📂</div>
                    <p>Click or Drag & Drop a handwriting image here</p>
                  </div>
                )}
              </label>
              <button className="btn-primary" onClick={predictImageFile} disabled={!selectedFile || loading}>
                {loading ? <div className="spinner"></div> : "Recognize Uploaded Image"}
              </button>
            </div>
          )}

          {/* Tab 3: Test Synthetic Text String */}
          {activeTab === 'synthetic' && (
            <div className="canvas-wrapper">
              <input
                type="text"
                className="input-text-area"
                value={synthText}
                onChange={(e) => setSynthText(e.target.value)}
                placeholder="Type Bangla text (e.g. বাংলাদেশ, বিজ্ঞান)"
              />
              <button className="btn-primary" onClick={predictSyntheticText} disabled={!synthText || loading}>
                {loading ? <div className="spinner"></div> : "Render & Test Recognition"}
              </button>
            </div>
          )}

          {error && (
            <div style={{ marginTop: '16px', color: '#ef4444', fontSize: '0.9rem', textAlign: 'center' }}>
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Prediction Results Card */}
        <div className="card output-card">
          <div className="card-title">🔍 HTR Prediction Results</div>

          {prediction ? (
            <>
              {/* Primary Result Hero */}
              <div className="result-hero">
                <div className="result-hero-label">Recognized Bangla Text</div>
                <div className="result-text-main">{prediction.corrected_prediction || "—"}</div>
                
                <div className={`dict-badge ${prediction.is_in_dictionary ? 'valid' : 'invalid'}`}>
                  {prediction.is_in_dictionary ? '✓ Trie Verified Bangla Word' : '⚠ Out of Vocabulary Candidate'}
                </div>
              </div>

              {/* Confidence Score Bar */}
              <div className="confidence-meter">
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '6px', fontWeight: 600 }}>
                  <span>Model Confidence Score</span>
                  <span>{prediction.confidence_score}%</span>
                </div>
                <div className="confidence-bar-bg">
                  <div
                    className="confidence-bar-fill"
                    style={{ width: `${Math.max(prediction.confidence_score, 5)}%` }}
                  />
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="metrics-row">
                <div className="metric-box">
                  <div className="metric-title">Raw Model Decoding</div>
                  <div className="metric-value">{prediction.raw_prediction || "—"}</div>
                </div>

                <div className="metric-box">
                  <div className="metric-title">Trie Corrected Result</div>
                  <div className="metric-value">{prediction.corrected_prediction || "—"}</div>
                </div>
              </div>
            </>
          ) : (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '60px 20px' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>✨</div>
              <p>Draw handwriting on the canvas or upload an image to view OCR predictions in real-time.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
