import React, { useState } from "react";
import { ImageTab } from "./tabs/ImageTab.jsx";
import { TextTab } from "./tabs/TextTab.jsx";

export default function App() {
  const [tab, setTab] = useState("image");

  return (
    <div className="demo-shell">
      {/* Header */}
      <header className="demo-header">
        <a className="demo-logo" href="#">
          <span>[Course Code]</span> — Deep Learning
        </a>
        <nav className="demo-nav" aria-label="Course links">
          <a href="../index.html">← Course Hub</a>
          <a href="../content.html">Technical Report</a>
          <a href="https://github.com/YOUR_USERNAME/YOUR_REPO" target="_blank" rel="noreferrer">
            GitHub ↗
          </a>
        </nav>
      </header>

      {/* Main Container */}
      <main className="demo-page">
        <div className="demo-hero-tag">Assignment Demonstration · React &amp; Vite</div>
        <h1 style={{ margin: "0 0 12px", fontFamily: "'DM Serif Display', Georgia, serif", fontSize: "36px" }}>
          Live <em>Inference</em> Demo
        </h1>
        <p style={{ color: "var(--ink2)", marginBottom: "28px" }}>
          Interactive playground testing checkpoints on sample inputs or custom user uploads.
        </p>

        {/* Tab Selection */}
        <div className="demo-tabs" role="tablist">
          <button
            type="button"
            className={tab === "image" ? "demo-tab demo-tab--active" : "demo-tab"}
            onClick={() => setTab("image")}
          >
            🖼 Image Classification
          </button>
          <button
            type="button"
            className={tab === "text" ? "demo-tab demo-tab--active" : "demo-tab"}
            onClick={() => setTab("text")}
          >
            📝 Text Classification
          </button>
        </div>

        {tab === "image" && <ImageTab />}
        {tab === "text" && <TextTab />}
      </main>

      {/* Footer */}
      <footer className="demo-footer">
        [Course Code] · [Group Name] · Live Demo Application &nbsp;·&nbsp; Instructor: [Instructor Name]
      </footer>
    </div>
  );
}
