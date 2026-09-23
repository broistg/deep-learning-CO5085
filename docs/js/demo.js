/**
 * Interactive Demo Logic for Deep Learning Course Template
 * Handles:
 *  1. Tab switching (Image, Text, Custom)
 *  2. File upload & drag-and-drop preview
 *  3. Sample selector buttons
 *  4. Mock or real API inference calls
 *  5. Animated Top-K probability bar chart rendering
 */

document.addEventListener('DOMContentLoaded', () => {
  // Tab Elements
  const tabs = document.querySelectorAll('.demo-tab');
  const taskTitle = document.getElementById('task-title');
  const taskDesc = document.getElementById('task-desc');

  // Input & Dropzone Elements
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const previewContainer = document.getElementById('preview-container');
  const previewImage = document.getElementById('preview-image');
  const clearPreviewBtn = document.getElementById('clear-preview-btn');
  const sampleButtons = document.querySelectorAll('.sample-btn');

  // Prediction Controls & Output
  const predictBtn = document.getElementById('predict-btn');
  const modelSelect = document.getElementById('model-select');
  const resultsContainer = document.getElementById('results-container');
  const latencyBadge = document.getElementById('latency-badge');
  const topkList = document.getElementById('topk-list');

  let currentFile = null;
  let currentTab = 'image';

  // --- 1. Tab Switching ---
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('demo-tab--active'));
      tab.classList.add('demo-tab--active');
      currentTab = tab.dataset.tab;

      if (currentTab === 'image') {
        taskTitle.textContent = 'Image Classification Task';
        taskDesc.textContent = 'Upload an image or select a sample below to inspect predictions.';
        document.getElementById('image-input-section').style.display = 'block';
        document.getElementById('text-input-section').style.display = 'none';
      } else if (currentTab === 'text') {
        taskTitle.textContent = 'Text Classification / NLP Task';
        taskDesc.textContent = 'Enter text below or pick a sample headline to evaluate sentiment or topic.';
        document.getElementById('image-input-section').style.display = 'none';
        document.getElementById('text-input-section').style.display = 'block';
      } else {
        taskTitle.textContent = 'Multimodal / Custom Task';
        taskDesc.textContent = 'Combine image and textual prompt to evaluate multimodal representations.';
        document.getElementById('image-input-section').style.display = 'block';
        document.getElementById('text-input-section').style.display = 'block';
      }

      // Hide previous results on tab switch
      resultsContainer.classList.remove('visible');
    });
  });

  // --- 2. Image File Upload & Preview ---
  if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
      dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFile(e.target.files[0]);
      }
    });
  }

  function handleFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file (PNG, JPG, WEBP).');
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      previewContainer.style.display = 'block';
      dropZone.style.display = 'none';
    };
    reader.readAsDataURL(file);
  }

  if (clearPreviewBtn) {
    clearPreviewBtn.addEventListener('click', () => {
      currentFile = null;
      previewImage.src = '';
      previewContainer.style.display = 'none';
      dropZone.style.display = 'block';
      resultsContainer.classList.remove('visible');
      if (fileInput) fileInput.value = '';
    });
  }

  // --- 3. Sample Preset Buttons ---
  sampleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const sampleUrl = btn.dataset.sampleUrl;
      previewImage.src = sampleUrl;
      previewContainer.style.display = 'block';
      dropZone.style.display = 'none';
      currentFile = { name: btn.textContent.trim(), isSample: true };
    });
  });

  // Text sample buttons
  const textSampleButtons = document.querySelectorAll('.text-sample-btn');
  const textInputArea = document.getElementById('text-input-area');
  textSampleButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      if (textInputArea) {
        textInputArea.value = btn.dataset.textSample;
      }
    });
  });

  // --- 4. Prediction Execution (Simulated or Real API) ---
  if (predictBtn) {
    predictBtn.addEventListener('click', async () => {
      // Validate input based on active tab
      if (currentTab === 'image' && !currentFile && previewContainer.style.display === 'none') {
        alert('Please select or upload an image first!');
        return;
      }

      if (currentTab === 'text' && (!textInputArea || !textInputArea.value.trim())) {
        alert('Please input some text or click a sample headline!');
        return;
      }

      // UI Loading state
      predictBtn.disabled = true;
      predictBtn.textContent = 'Running Inference...';
      const startTime = performance.now();

      try {
        /**
         * =========================================================================
         * BACKEND INTEGRATION NOTE:
         * To connect to your FastAPI / Flask backend, replace the mock block below:
         * 
         * const formData = new FormData();
         * formData.append('file', currentFile);
         * formData.append('model', modelSelect.value);
         * const response = await fetch('http://localhost:8000/api/predict', {
         *   method: 'POST',
         *   body: formData
         * });
         * const data = await response.json();
         * renderResults(data.predictions, data.latency_ms);
         * =========================================================================
         */

        // Simulated network latency (250ms - 450ms)
        await new Promise(resolve => setTimeout(resolve, 320));
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);

        // Generate mock predictions according to task
        let mockPredictions = [];
        if (currentTab === 'image') {
          mockPredictions = [
            { label: 'Pine Tree', confidence: 0.842 },
            { label: 'Oak Tree', confidence: 0.089 },
            { label: 'Willow', confidence: 0.035 },
            { label: 'Palm Tree', confidence: 0.021 },
            { label: 'Maple', confidence: 0.013 }
          ];
        } else if (currentTab === 'text') {
          mockPredictions = [
            { label: 'Technology / AI', confidence: 0.912 },
            { label: 'Business & Finance', confidence: 0.054 },
            { label: 'Science & Education', confidence: 0.022 },
            { label: 'Politics', confidence: 0.008 },
            { label: 'Entertainment', confidence: 0.004 }
          ];
        } else {
          mockPredictions = [
            { label: 'High Alignment (Image + Text)', confidence: 0.785 },
            { label: 'Moderate Context Overlap', confidence: 0.162 },
            { label: 'Unrelated Modalities', confidence: 0.053 }
          ];
        }

        renderResults(mockPredictions, latency);

      } catch (err) {
        console.error('Inference error:', err);
        alert('Inference error. Check console for details.');
      } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = '⚡ Run Prediction';
      }
    });
  }

  // --- 5. Render Top-K Horizontal Probability Bars ---
  function renderResults(predictions, latency) {
    if (latencyBadge) {
      latencyBadge.textContent = `${latency} ms`;
    }

    if (!topkList) return;
    topkList.innerHTML = '';

    predictions.forEach((item, idx) => {
      const percentage = (item.confidence * 100).toFixed(1);
      const isTop1 = idx === 0;

      const itemDiv = document.createElement('div');
      itemDiv.className = 'topk-item';

      itemDiv.innerHTML = `
        <div class="topk-label-row">
          <span style="${isTop1 ? 'font-weight: 600; color: var(--accent);' : 'color: var(--ink);'}">
            ${idx + 1}. ${item.label} ${isTop1 ? '★' : ''}
          </span>
          <span style="color: ${isTop1 ? 'var(--accent)' : 'var(--ink2)'}; font-weight: 500;">
            ${percentage}%
          </span>
        </div>
        <div class="topk-bar-bg">
          <div class="topk-bar-fill" style="width: 0%; background: ${isTop1 ? 'var(--accent)' : 'var(--ink2)'};"></div>
        </div>
      `;

      topkList.appendChild(itemDiv);

      // Trigger CSS width animation after mounting
      setTimeout(() => {
        const fill = itemDiv.querySelector('.topk-bar-fill');
        if (fill) fill.style.width = `${percentage}%`;
      }, 50);
    });

    resultsContainer.classList.add('visible');
  }
});
