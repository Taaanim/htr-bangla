import React, { useState, useRef, useEffect } from 'react';
import './index.css';

const API_BASE = 'http://localhost:5001/api';
const confColor = (c) => (c >= 90 ? '#10b981' : c >= 70 ? '#f59e0b' : '#ef4444');
const roundAcc = (val) => (isNaN(val) ? 0 : Math.round(val));

export default function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' or 'canvas'
  const [ocrMode, setOcrMode] = useState('char');       // 'char' or 'document'
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

  const getCroppedCanvas = (canvas) => {
    if (!canvas) return canvas;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    const imgData = ctx.getImageData(0, 0, width, height);
    const data = imgData.data;

    let minX = width, minY = height, maxX = 0, maxY = 0;
    let found = false;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = (y * width + x) * 4;
        const r = data[i], g = data[i + 1], b = data[i + 2];
        if (r < 240 || g < 240 || b < 240) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
          found = true;
        }
      }
    }

    if (!found) return canvas;

    const pad = 6;
    minX = Math.max(0, minX - pad);
    minY = Math.max(0, minY - pad);
    maxX = Math.min(width, maxX + pad);
    maxY = Math.min(height, maxY + pad);

    const cropW = maxX - minX;
    const cropH = maxY - minY;

    const croppedCanvas = document.createElement('canvas');
    croppedCanvas.width = cropW;
    croppedCanvas.height = cropH;
    const croppedCtx = croppedCanvas.getContext('2d');
    croppedCtx.fillStyle = '#ffffff';
    croppedCtx.fillRect(0, 0, cropW, cropH);
    croppedCtx.drawImage(canvas, minX, minY, cropW, cropH, 0, 0, cropW, cropH);

    return croppedCanvas;
  };

  const predictCanvas = async () => {
    if (!canvasRef.current) return;
    const croppedCanvas = getCroppedCanvas(canvasRef.current);
    const imageBase64 = croppedCanvas.toDataURL('image/png');
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

  const [syntheticText, setSyntheticText] = useState('আমার সোনার বাংলা, আমি তোমায় ভালোবাসি।\nচিরদিন তোমার আকাশ, তোমার বাতাস, আমার প্রাণে বাজায় বাঁশি॥');
  const [isPredictingSynth, setIsPredictingSynth] = useState(false);
  const [synthResults, setSynthResults] = useState(null);

  const predictSyntheticText = async () => {
    if (!syntheticText || !syntheticText.trim()) return;
    setIsPredictingSynth(true); setSynthResults(null); setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict/synthetic_text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: syntheticText })
      });
      const data = await res.json();
      if (data.success) {
        setSynthResults(data.results);
      } else {
        setError(data.error || "Synthetic text prediction failed");
      }
    } catch {
      setError("Cannot connect to API server on localhost:5001");
    } finally {
      setIsPredictingSynth(false);
    }
  };

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
            <button className={`tab-btn ${activeTab === 'synthetic' ? 'active' : ''}`}
              onClick={() => setActiveTab('synthetic')}>📝 Predict Synthetic Text</button>
          </div>

          {activeTab === 'synthetic' && (
            <div className="canvas-wrapper" style={{ marginTop: '12px' }}>
              <div style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '6px' }}>
                Paste or type custom Bangla text to generate synthetic handwriting & evaluate trained model:
              </div>
              <textarea
                value={syntheticText}
                onChange={(e) => setSyntheticText(e.target.value)}
                placeholder="বাংলা টেক্সট লিখুন বা পেস্ট করুন (যেমন: আমার সোনার বাংলা...)"
                rows={5}
                style={{
                  width: '100%',
                  background: '#0d1117',
                  color: '#38bdf8',
                  border: '1px solid #1e293b',
                  borderRadius: '8px',
                  padding: '12px',
                  fontFamily: 'monospace, serif',
                  fontSize: '1rem',
                  lineHeight: '1.6',
                  resize: 'vertical',
                  boxSizing: 'border-box'
                }}
              />
              <button
                className="btn-primary"
                onClick={predictSyntheticText}
                disabled={isPredictingSynth || !syntheticText.trim()}
                style={{ marginTop: '12px', width: '100%' }}>
                {isPredictingSynth ? <div className="spinner"></div> : "🔍 Predict Custom Text With Trained Model"}
              </button>
            </div>
          )}

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
          <div className="card-title">🔍 Recognized Output & Benchmark Results</div>
          
          {synthResults ? (
            <div>
              <div style={{ display: 'flex', gap: '8px', marginBottom: '14px' }}>
                <span style={{ fontSize: '0.8rem', background: 'rgba(16, 185, 129, 0.15)', color: '#10b981', padding: '4px 10px', borderRadius: '6px', border: '1px solid #10b981', fontWeight: 700 }}>
                  Exact Matches: {synthResults.filter(r => r.match).length} / {synthResults.length}
                </span>
                <span style={{ fontSize: '0.8rem', background: 'rgba(59, 130, 246, 0.15)', color: '#3b82f6', padding: '4px 10px', borderRadius: '6px', border: '1px solid #3b82f6', fontWeight: 700 }}>
                  Sequence Accuracy: {roundAcc((synthResults.filter(r => r.match).length / synthResults.length) * 100)}%
                </span>
              </div>

              <div style={{ maxHeight: '420px', overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '10px', paddingRight: '4px' }}>
                {synthResults.map((r, idx) => (
                  <div key={idx} style={{
                    background: 'var(--bg-main)',
                    border: r.match ? '1.5px solid #10b981' : '1.5px solid #ef4444',
                    borderRadius: '10px',
                    padding: '10px 14px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    boxShadow: r.match ? '0 2px 8px rgba(16,185,129,0.1)' : '0 2px 8px rgba(239,68,68,0.1)'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
                      {r.img_b64 ? (
                        <img src={r.img_b64} alt={r.gt} style={{ height: '52px', minWidth: '60px', objectFit: 'contain', borderRadius: '6px', background: '#000', border: '1.5px solid #38bdf8', imageRendering: 'crisp-edges' }} />
                      ) : (
                        <div style={{ height: '52px', width: '60px', background: '#000', borderRadius: '6px' }}></div>
                      )}
                      <div>
                        <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Target: <strong style={{ color: '#38bdf8', fontSize: '1.05rem', fontWeight: 700 }}>{r.gt}</strong></div>
                        <div style={{ fontSize: '1.25rem', fontWeight: 700, color: r.match ? '#10b981' : '#f87171', marginTop: '2px' }}>{r.prediction || '(no pred)'}</div>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '0.85rem', color: r.match ? '#10b981' : '#ef4444', fontWeight: 700 }}>
                        {r.match ? "✓ Match" : "✕ Diff"}
                      </div>
                      <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '2px' }}>{r.confidence}%</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : prediction ? (
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
                  {typeof prediction === 'string'
                    ? prediction
                    : (typeof prediction?.prediction === 'string'
                        ? prediction.prediction
                        : (typeof prediction?.prediction?.prediction === 'string'
                            ? prediction.prediction.prediction
                            : JSON.stringify(prediction?.prediction || "(No text recognized)")))}
                </div>
              </div>

              {/* Model Input Image Preview Card */}
              {prediction.img_b64 && (
                <div style={{
                  marginBottom: '16px',
                  background: '#0d1117',
                  border: '1.5px solid #38bdf8',
                  borderRadius: '10px',
                  padding: '12px 16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '16px',
                  boxShadow: '0 4px 12px rgba(56, 189, 248, 0.15)'
                }}>
                  <img
                    src={prediction.img_b64}
                    alt="Model Input Patch Preview"
                    style={{
                      height: '64px',
                      width: '64px',
                      objectFit: 'contain',
                      borderRadius: '8px',
                      background: '#000',
                      border: '1px solid #38bdf8',
                      imageRendering: 'crisp-edges'
                    }}
                  />
                  <div>
                    <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      🖼️ Model Input Image Patch (32×32 Preprocessed Tensor)
                    </div>
                    <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#ffffff', marginTop: '2px' }}>
                      {typeof prediction.prediction === 'string' ? prediction.prediction : (typeof prediction === 'string' ? prediction : '')}
                    </div>
                    <div style={{ fontSize: '0.85rem', color: confColor(prediction.confidence), fontWeight: 700 }}>
                      {prediction.confidence}% Confidence
                    </div>
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
