import React, { useState, useRef, useEffect } from 'react';
import './index.css';

const API_BASE = 'http://localhost:5001/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' or 'canvas'
  const [ocrMode, setOcrMode] = useState('document');   // 'document' or 'char'
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [selectedCharDetail, setSelectedCharDetail] = useState(null);

  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushWidth, setBrushWidth] = useState(6);

  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);

  useEffect(() => { fetchHealth(); }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      setHealth(await res.json());
    } catch { setHealth({ status: "connecting" }); }
  };

  useEffect(() => {
    if (activeTab === 'canvas' && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, [activeTab]);

  const getPos = (e) => {
    if (!canvasRef.current) return { x: 0, y: 0 };
    const rect = canvasRef.current.getBoundingClientRect();
    const clientX = e.touches && e.touches[0] ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches && e.touches[0] ? e.touches[0].clientY : e.clientY;

    const scaleX = canvasRef.current.width / rect.width;
    const scaleY = canvasRef.current.height / rect.height;

    const x = (clientX - rect.left) * scaleX;
    const y = (clientY - rect.top) * scaleY;
    return { x, y };
  };

  const startDrawing = (e) => {
    if (e.type === 'touchstart') e.preventDefault();
    const ctx = canvasRef.current.getContext('2d');
    const { x, y } = getPos(e);
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
    if (e.type === 'touchmove') e.preventDefault();
    const ctx = canvasRef.current.getContext('2d');
    const { x, y } = getPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => setIsDrawing(false);

  const clearCanvas = () => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
    setPrediction(null); setError(null); setSelectedCharDetail(null);
  };

  const predictCanvas = async () => {
    const imageBase64 = canvasRef.current.toDataURL('image/png');
    setLoading(true); setError(null); setSelectedCharDetail(null);
    try {
      const res = await fetch(`${API_BASE}/predict/image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imageBase64, mode: ocrMode })
      });
      const data = await res.json();
      if (data.success) setPrediction(data.prediction);
      else setError(data.error || "Prediction failed");
    } catch { setError("Cannot connect to API server on localhost:5001"); }
    finally { setLoading(false); }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null); setError(null); setSelectedCharDetail(null);
    }
  };

  const predictFile = async () => {
    if (!selectedFile) return;
    setLoading(true); setError(null); setSelectedCharDetail(null);
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('mode', ocrMode);
    try {
      const res = await fetch(`${API_BASE}/predict/image`, { method: 'POST', body: formData });
      const data = await res.json();
      if (data.success) setPrediction(data.prediction);
      else setError(data.error || "Prediction failed");
    } catch { setError("Cannot connect to API server on localhost:5001"); }
    finally { setLoading(false); }
  };

  const confColor = (c) => c >= 90 ? '#10b981' : c >= 70 ? '#f59e0b' : '#ef4444';

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="badge-sota">⚡ Hierarchical Document Layout OCR</div>
        <h1>Bangla <span>Hierarchical Document OCR</span></h1>
        <p>Detect Blocks (Cyan), Lines (Blue), Words (Dark Blue), & Individual Symbols (Purple)</p>
      </header>

      <div className="status-banner">
        <div className="status-info">
          <div className="status-dot"></div>
          <span>Model Status: <strong>{health?.activeCheckpoint || "connecting..."}</strong></span>
        </div>
        <div className="status-chip">MPS GPU</div>
      </div>

      <div className="main-grid">
        <div className="card">
          <div className="card-title">✏️ Input & Recognition Mode</div>

          {/* OCR Mode Selector */}
          <div style={{ marginBottom: '16px', display: 'flex', gap: '8px' }}>
            <button
              className={`tab-btn ${ocrMode === 'document' ? 'active' : ''}`}
              style={{ flex: 1, padding: '8px 12px', fontSize: '0.85rem' }}
              onClick={() => { setOcrMode('document'); setPrediction(null); }}>
              📄 Hierarchical Paragraph / Document OCR
            </button>
            <button
              className={`tab-btn ${ocrMode === 'char' ? 'active' : ''}`}
              style={{ flex: 1, padding: '8px 12px', fontSize: '0.85rem' }}
              onClick={() => { setOcrMode('char'); setPrediction(null); }}>
              🔤 Single Character
            </button>
          </div>

          <div className="tabs-header">
            <button className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}>Upload Document</button>
            <button className={`tab-btn ${activeTab === 'canvas' ? 'active' : ''}`}
              onClick={() => setActiveTab('canvas')}>Draw Canvas</button>
          </div>

          {activeTab === 'upload' && (
            <div className="canvas-wrapper">
              <label className="dropzone" style={{ width: '100%', minHeight: '220px' }}>
                <input type="file" accept="image/*" onChange={handleFileChange} style={{ display: 'none' }} />
                {previewUrl
                  ? <img src={previewUrl} alt="Preview" className="preview-image" style={{ maxHeight: '250px', objectFit: 'contain' }} />
                  : <div style={{ textAlign: 'center' }}>
                      <div style={{ fontSize: '2.5rem', marginBottom: '8px' }}>📄</div>
                      <p style={{ fontWeight: 600, margin: '4px 0' }}>Drop Paragraph or Document Image</p>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Supports JPG, PNG, scanned handwriting</span>
                    </div>}
              </label>
              <button className="btn-primary" onClick={predictFile} disabled={!selectedFile || loading}>
                {loading ? <div className="spinner"></div> : (ocrMode === 'document' ? "Analyze Document Hierarchy" : "Recognize Character")}
              </button>
            </div>
          )}

          {activeTab === 'canvas' && (
            <div className="canvas-wrapper">
              <div className="canvas-container">
                <canvas ref={canvasRef} width={320} height={320}
                  onMouseDown={startDrawing} onMouseMove={draw}
                  onMouseUp={stopDrawing} onMouseLeave={stopDrawing}
                  onTouchStart={startDrawing} onTouchMove={draw} onTouchEnd={stopDrawing} />
              </div>
              <div className="canvas-tools">
                <div className="brush-slider">
                  <span>Stroke:</span>
                  <input type="range" min="2" max="20" value={brushWidth}
                    onChange={(e) => setBrushWidth(Number(e.target.value))} />
                  <span>{brushWidth}px</span>
                </div>
                <button className="btn-secondary" onClick={clearCanvas}>Clear</button>
              </div>
              <button className="btn-primary" onClick={predictCanvas} disabled={loading}>
                {loading ? <div className="spinner"></div> : "Recognize Canvas"}
              </button>
            </div>
          )}

          {error && <div style={{ marginTop: '16px', color: '#ef4444', textAlign: 'center' }}>⚠️ {error}</div>}
        </div>

        <div className="card output-card">
          <div className="card-title">🔍 Recognized Output & Hierarchy</div>
          {prediction ? (
            <>
              {/* Hierarchical Document Layout Badges */}
              {prediction.blocks && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginBottom: '12px' }}>
                  <span style={{ fontSize: '0.75rem', background: 'rgba(6, 182, 212, 0.15)', color: '#06b6d4', padding: '3px 8px', borderRadius: '6px', border: '1px solid #06b6d4', fontWeight: 600 }}>
                    Cyan: {prediction.blocks?.length || 1} Block(s)
                  </span>
                  <span style={{ fontSize: '0.75rem', background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6', padding: '3px 8px', borderRadius: '6px', border: '1px solid #3b82f6', fontWeight: 600 }}>
                    Blue: {prediction.num_lines || 1} Line(s)
                  </span>
                  <span style={{ fontSize: '0.75rem', background: 'rgba(29, 78, 216, 0.15)', color: '#60a5fa', padding: '3px 8px', borderRadius: '6px', border: '1px solid #2563eb', fontWeight: 600 }}>
                    Dark Blue: {prediction.num_words || 1} Word(s)
                  </span>
                  <span style={{ fontSize: '0.75rem', background: 'rgba(139, 92, 246, 0.15)', color: '#a78bfa', padding: '3px 8px', borderRadius: '6px', border: '1px solid #8b5cf6', fontWeight: 600 }}>
                    Purple: {prediction.num_chars || 1} Symbol(s)
                  </span>
                  {prediction.rotation !== undefined && (
                    <span style={{ fontSize: '0.75rem', background: 'var(--bg-main)', color: 'var(--text-muted)', padding: '3px 8px', borderRadius: '6px', border: '1px solid var(--border)', fontWeight: 600 }}>
                      Rotation: {prediction.rotation}°
                    </span>
                  )}
                </div>
              )}

              {/* Paragraph / Multi-line Text Output */}
              <div className="result-hero" style={{ textAlign: 'left', minHeight: '120px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                  <div className="result-hero-label">Recognized Text Output</div>
                </div>

                <div style={{
                  background: '#0d1117',
                  color: '#38bdf8',
                  padding: '14px 16px',
                  borderRadius: '8px',
                  fontFamily: 'monospace, serif',
                  fontSize: '1.25rem',
                  lineHeight: '1.8',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                  border: '1px solid #1e293b'
                }}>
                  {prediction.prediction || "(No text recognized)"}
                </div>
              </div>

              {/* Single Character Preprocessed Thumbnail (if char mode) */}
              {prediction.img_b64 && (
                <div style={{ marginTop: '16px', display: 'flex', alignItems: 'center', gap: '16px', background: 'var(--bg-main)', padding: '12px', borderRadius: '12px' }}>
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '4px' }}>Model Input Patch (32×32)</div>
                    <img src={prediction.img_b64} alt="Preprocessed" style={{ width: '64px', height: '64px', imageRendering: 'pixelated', border: '1px solid var(--border)', borderRadius: '6px' }} />
                  </div>
                  <div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Top Class</div>
                    <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>{prediction.prediction}</div>
                    <div style={{ fontSize: '0.85rem', color: confColor(prediction.confidence), fontWeight: 600 }}>{prediction.confidence}% confidence</div>
                  </div>
                </div>
              )}

              {/* Character Inspection Grid */}
              {prediction.characters && prediction.characters.length > 0 && (
                <div style={{ marginTop: '20px' }}>
                  <div style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '10px' }}>
                    🖼️ Detected Symbols ({prediction.characters.length})
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(68px, 1fr))', gap: '8px', maxHeight: '220px', overflowY: 'auto', paddingRight: '4px' }}>
                    {prediction.characters.map((c, i) => (
                      <div
                        key={i}
                        onClick={() => setSelectedCharDetail(c)}
                        style={{
                          background: selectedCharDetail === c ? 'var(--accent-bg)' : 'var(--bg-main)',
                          border: selectedCharDetail === c ? '2px solid var(--primary)' : '1px solid var(--border)',
                          borderRadius: '8px',
                          padding: '6px',
                          textAlign: 'center',
                          cursor: 'pointer',
                          transition: 'all 0.15s ease'
                        }}>
                        {c.img_b64 ? (
                          <img src={c.img_b64} alt={c.symbol || c.char} style={{ width: '36px', height: '36px', imageRendering: 'pixelated', borderRadius: '4px', background: '#000' }} />
                        ) : (
                          <div style={{ height: '36px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700 }}>␣</div>
                        )}
                        <div style={{ fontSize: '0.95rem', fontWeight: 700, marginTop: '2px' }}>{(c.symbol || c.char) === ' ' ? 'space' : (c.symbol || c.char)}</div>
                        <div style={{ fontSize: '0.65rem', color: confColor(c.confidence), fontWeight: 600 }}>{c.confidence}%</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Selected Symbol Detail Inspector */}
              {selectedCharDetail && (
                <div style={{ marginTop: '16px', background: '#0f172a', color: '#f8fafc', padding: '14px', borderRadius: '10px', border: '1px solid #334155' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <div style={{ fontSize: '0.85rem', fontWeight: 700, color: '#38bdf8' }}>🔍 Symbol Inspector & Bounding Box</div>
                    <button onClick={() => setSelectedCharDetail(null)} style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer' }}>✕</button>
                  </div>
                  <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
                    {selectedCharDetail.img_b64 && (
                      <img src={selectedCharDetail.img_b64} alt="Detail" style={{ width: '56px', height: '56px', imageRendering: 'pixelated', border: '1px solid #475569', borderRadius: '6px' }} />
                    )}
                    <div>
                      <div style={{ fontSize: '1.2rem', fontWeight: 700 }}>Symbol: <span style={{ color: '#38bdf8' }}>{selectedCharDetail.symbol || selectedCharDetail.char}</span></div>
                      <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Line {selectedCharDetail.line + 1} • BBox: [{selectedCharDetail.bbox?.join(', ')}]</div>
                      {selectedCharDetail.corner_points && (
                        <div style={{ fontSize: '0.75rem', color: '#64748b', fontFamily: 'monospace' }}>
                          Corners: {JSON.stringify(selectedCharDetail.corner_points)}
                        </div>
                      )}
                      <div style={{ fontSize: '0.85rem', color: confColor(selectedCharDetail.confidence), fontWeight: 600 }}>Confidence: {selectedCharDetail.confidence}%</div>
                    </div>
                  </div>
                  {selectedCharDetail.top5 && selectedCharDetail.top5.length > 0 && (
                    <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px solid #334155' }}>
                      <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>Top 5 Candidates for this symbol patch:</div>
                      <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                        {selectedCharDetail.top5.map((candidate, idx) => (
                          <span key={idx} style={{ fontSize: '0.75rem', background: '#1e293b', padding: '2px 8px', borderRadius: '4px', border: '1px solid #334155' }}>
                            {candidate.label}: {candidate.confidence}%
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          ) : (
            <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '60px 20px' }}>
              <div style={{ fontSize: '2.5rem', marginBottom: '12px' }}>🔍</div>
              <p>Upload a document to detect Blocks (Cyan), Lines (Blue), Words (Dark Blue), and Symbols (Purple).</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
