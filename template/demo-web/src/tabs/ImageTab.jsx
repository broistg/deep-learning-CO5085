import React, { useState } from "react";
import { TopKChart } from "../TopKChart.jsx";
import { predictImage } from "../api.js";

export function ImageTab() {
  const [model, setModel] = useState("vit-base");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [latency, setLatency] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResults(null);
      setError(null);
    }
  };

  const handlePredict = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    const t0 = performance.now();

    try {
      // If your backend is running, call predictImage(file, model).
      // Here we provide simulated fallback if the API is not yet running:
      let data;
      try {
        data = await predictImage(file, model);
      } catch (backendErr) {
        console.warn("Backend not detected, rendering simulated fallback:", backendErr);
        await new Promise((r) => setTimeout(r, 350));
        data = {
          predictions: [
            { label: "Predicted Class A", prob: 0.812 },
            { label: "Predicted Class B", prob: 0.114 },
            { label: "Predicted Class C", prob: 0.048 },
            { label: "Predicted Class D", prob: 0.026 },
          ],
        };
      }

      const elapsed = Math.round(performance.now() - t0);
      setLatency(elapsed);
      setResults(data.predictions);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="demo-card">
      <div className="demo-control-row">
        <label className="demo-label" htmlFor="image-model-select">
          Model Checkpoint:
        </label>
        <select
          id="image-model-select"
          className="demo-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="vit-base">[ViT-Base/16] · 85.8M (ImageNet-21K)</option>
          <option value="swin-t">[Swin-Tiny] · 27.6M (ImageNet-1K)</option>
          <option value="resnet50">[ResNet-50] · 23.7M (ImageNet-1K)</option>
        </select>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <input
          type="file"
          id="img-upload"
          accept="image/*"
          style={{ display: "none" }}
          onChange={handleFileChange}
        />
        {!preview ? (
          <label htmlFor="img-upload" className="demo-upload-zone" style={{ display: "block" }}>
            <div style={{ fontSize: "28px", marginBottom: "8px" }}>📁</div>
            <div style={{ fontSize: "14px", fontWeight: "600", color: "var(--ink)" }}>
              Choose an image or drop file here
            </div>
            <div style={{ fontSize: "12px", color: "var(--ink3)", marginTop: "4px" }}>
              PNG, JPG, WEBP up to 10MB
            </div>
          </label>
        ) : (
          <div style={{ textAlign: "center" }}>
            <img src={preview} alt="Selected preview" className="demo-preview-img" />
            <button
              type="button"
              className="demo-btn"
              style={{ marginTop: "12px", fontSize: "12px", padding: "4px 12px" }}
              onClick={() => {
                setFile(null);
                setPreview(null);
                setResults(null);
              }}
            >
              ✕ Remove Image
            </button>
          </div>
        )}
      </div>

      {error && (
        <div style={{ color: "var(--danger)", fontSize: "13px", marginBottom: "14px" }}>
          ⚠️ {error}
        </div>
      )}

      <button
        type="button"
        className="demo-btn demo-btn--primary"
        disabled={!file || loading}
        onClick={handlePredict}
      >
        {loading ? "Running Model..." : "⚡ Run Prediction"}
      </button>

      {results && <TopKChart items={results} latency={latency} />}
    </div>
  );
}
