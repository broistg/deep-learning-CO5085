import React from "react";

export function TopKChart({ items = [], latency = null }) {
  if (!items || items.length === 0) return null;

  return (
    <div style={{ marginTop: "24px", paddingTop: "20px", borderTop: "1px solid var(--border)" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
        <h3 style={{ margin: 0, fontSize: "16px", fontWeight: "600", color: "var(--ink)" }}>
          Top Predictions
        </h3>
        {latency != null && (
          <span
            style={{
              fontSize: "12px",
              padding: "3px 10px",
              borderRadius: "999px",
              background: "#E1F5EE",
              color: "#085041",
              fontWeight: "600",
            }}
          >
            {latency} ms
          </span>
        )}
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {items.map((item, idx) => {
          const pct = ((item.prob || item.confidence || 0) * 100).toFixed(1);
          const isTop = idx === 0;

          return (
            <div key={item.label || idx}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: "13px",
                  fontWeight: isTop ? "600" : "500",
                  color: isTop ? "var(--accent)" : "var(--ink)",
                  marginBottom: "4px",
                }}
              >
                <span>
                  {idx + 1}. {item.label} {isTop && "★"}
                </span>
                <span>{pct}%</span>
              </div>
              <div
                style={{
                  height: "8px",
                  background: "var(--bg2)",
                  borderRadius: "4px",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    height: "100%",
                    width: `${pct}%`,
                    background: isTop ? "var(--accent)" : "var(--ink2)",
                    borderRadius: "4px",
                    transition: "width 0.4s ease",
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
