# Deep Learning Course Project — Frontend Template

An editorial, publication-grade frontend template designed for Deep Learning and Computer Vision course projects. Features an academic warm-parchment aesthetic, sophisticated typography (`DM Serif Display` + `DM Sans`), high-density benchmark tables, architecture cards, figure grids, and interactive demo playgrounds.

---

## 📁 Directory Structure

```
template/
├── index.html                   # 🏠 Main Course Portal & Hub (Hero, Team, Deliverables, Links)
├── content.html                 # 📄 Assignment Technical Report (EDA, Models, Tables, Figures, Ablations)
├── demo.html                    # ⚡ Standalone Live Demo (Vanilla JS: Tabs, File Drop, Top-K Bar Chart)
├── css/
│   └── style.css                # 🎨 Unified CSS Design System (Color tokens, cards, badges, responsive layout)
├── js/
│   └── demo.js                  # 🧠 Interactive client logic for demo.html (Previews, mock/real inference)
├── assets/
│   ├── placeholder-figure.svg   # 📊 Vector SVG placeholder for confusion matrices and loss curves
│   └── placeholder-sample.svg   # 🖼 Vector SVG placeholder for dataset inspection grids
├── demo-web/                    # ⚛️ Optional React + Vite template for full-stack interactive demos
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── App.jsx
│       ├── index.css
│       ├── TopKChart.jsx
│       ├── api.js
│       └── tabs/
│           ├── ImageTab.jsx
│           └── TextTab.jsx
└── README.md                    # 📖 Documentation & customization guide (this file)
```

---

## 🚀 Quick Start

### 1. Static HTML Sites (Zero Installation Required)
You can directly open any of the HTML pages in your browser (Chrome, Edge, Firefox, Safari):
- Double-click **`index.html`** to view the main course hub.
- Double-click **`content.html`** to view the technical report template.
- Double-click **`demo.html`** to test the interactive model playground.

### 2. React + Vite Demo App (Optional Full-Stack Setup)
If your team wants to develop an interactive web app with React:
```bash
cd demo-web
npm install
npm run dev
```
The app will launch at `http://localhost:5173`. Any requests to `/api/*` are proxied to `http://127.0.0.1:8000` (FastAPI / Flask).

---

## 🎨 Design System & Styling Customization

All styling tokens are centralized in [`css/style.css`](css/style.css). You can re-theme the entire project in seconds by adjusting the `:root` variables:

```css
:root {
  /* Colors */
  --bg: #F9F7F4;                 /* Warm off-white paper background */
  --bg2: #F0EDE8;                /* Card & table header tint */
  --ink: #1A1814;                /* Primary deep charcoal text */
  --ink2: #6B6760;               /* Secondary text */
  --ink3: #A8A49F;               /* Subtle captions & labels */
  --accent: #C5440A;             /* Signature terracotta accent */
  --accent2: #E8593C;            /* Coral hover & avatar background */
  --border: rgba(26, 24, 20, 0.10);

  /* Typography */
  /* Headlines: 'DM Serif Display', Georgia, serif */
  /* Body/UI: 'DM Sans', sans-serif */
}
```

---

## ✍️ How to Customize for Your Project

### 1. Main Hub (`index.html`)
- Search for `[Course Code]` and replace with your course ID (e.g. `CS231n`, `CO3133`, `EECS 498`).
- Update the team members in `<div class="members-grid">` with your team's names, student IDs, and roles.
- Customize the assignment cards in `<div class="assignments">` with your deliverable title, description, and links to your report, demo, and slide deck.

### 2. Technical Report (`content.html`)
- **Part Navigation (`.assignment-parts`)**: If your assignment has multiple sub-tasks (e.g., Image, Text, Multimodal), duplicate links or set `class="current"`.
- **Dataset / EDA (`#dataset`)**: Fill in dataset size, resolution, splits, and sample figures (`assets/placeholder-sample.svg`).
- **Model Architectures (`#architectures`)**: Edit the `.model-card` blocks with your model backbones (e.g., ResNet, ViT, YOLO, BERT), parameter counts, and test accuracies.
- **Results Table (`#results`)**: Update the `.results-table` with your validation and test accuracies, macro-F1, and training wall times.
- **Error Analysis (`#analysis`)**: Replace `assets/placeholder-figure.svg` with your actual confusion matrices, t-SNE projections, or loss plots.

### 3. Live Demo (`demo.html`)
- The standalone demo includes working client-side previews, sample buttons, and simulated Top-K probability bars.
- To connect it to your live PyTorch/FastAPI backend, open [`js/demo.js`](js/demo.js) and replace the mock block with your real `fetch()` call:
```javascript
const formData = new FormData();
formData.append('file', currentFile);
formData.append('model', modelSelect.value);

const res = await fetch('http://localhost:8000/api/predict', {
  method: 'POST',
  body: formData
});
const data = await res.json();
renderResults(data.predictions, data.latency_ms);
```

---

## 🏷 Badges & Utility Classes

The design system includes pre-configured badges for popular model families and task states:

| Class | Appearance | Usage |
| :--- | :--- | :--- |
| `.badge-done` | Light Green | Completed milestones |
| `.badge-wip` | Warm Amber | In-progress assignments |
| `.badge-todo` | Muted Gray | Upcoming / Planned tasks |
| `.badge-cnn` | Soft Violet | Convolutional backbones (ResNet, EfficientNet, ConvNeXt) |
| `.badge-vit` | Soft Emerald | Vision Transformers (ViT, Swin, DeiT) |
| `.badge-rnn` | Soft Terracotta | Recurrent networks (LSTM, GRU) |
| `.badge-transformer` | Soft Blue | NLP Transformers (BERT, RoBERTa, GPT) |

---

## 📱 Responsive Layouts

All pages are fully responsive out-of-the box:
- **Desktop**: Multi-column grids, side-by-side model comparisons, full-width benchmark tables with sticky frosted headers.
- **Tablet / Mobile**: Automatically collapses grids into stacked single-column views with smooth scrolling and finger-friendly tap targets.
