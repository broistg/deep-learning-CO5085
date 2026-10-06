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

  const modelConfigs = {
    image: [
      { value: 'vit-base', label: '[ViT-Base/16] · 85.8M (Vision Transformer - A1)' },
      { value: 'resnet50', label: '[ResNet-50] · 23.7M (CNN Backbone - A1)' },
      { value: 'custom-cnn', label: '[Custom CNN] · 1.2M (E1 Baseline)' },
      { value: 'mlp', label: '[MLP Classifier] · 0.2M (E1)' },
      { value: 'softmax', label: '[Softmax Linear] · 7.8K (E1)' }
    ],
    sequence: [
      { value: 'lstm-row', label: '[Image-LSTM (Row Scan)] · 28 timesteps (E3)' },
      { value: 'gru-row', label: '[Image-GRU (Row Scan)] · 28 timesteps (E3)' },
      { value: 'lstm-patch', label: '[Image-LSTM (Patch Seq)] · 16 patches (E3)' },
      { value: 'gru-patch', label: '[Image-GRU (Patch Seq)] · 16 patches (E3)' },
      { value: 'e1-baseline-cnn', label: '[E1 CNN Baseline] · Đối chứng không hồi quy' }
    ],
    attention: [
      { value: 'scratch-msa-patch4', label: '[Scratch MSA (Patch 4×4)] · Tự viết từ đầu (E2)' },
      { value: 'pytorch-msa-patch4', label: '[PyTorch nn.MultiheadAttention] · Đối chứng chuẩn (E2)' },
      { value: 'scratch-msa-stem', label: '[Scratch MSA (CNN-Stem)] · Đặc trưng lai (E2)' },
      { value: 'scratch-msa-row', label: '[Scratch MSA (Row Tokens)] · Quét theo hàng (E2)' }
    ]
  };

  function updateModelOptions(tabKey) {
    if (!modelSelect) return;
    modelSelect.innerHTML = '';
    const options = modelConfigs[tabKey] || modelConfigs.image;
    options.forEach(opt => {
      const optionElem = document.createElement('option');
      optionElem.value = opt.value;
      optionElem.textContent = opt.label;
      modelSelect.appendChild(optionElem);
    });
  }

  // --- 1. Tab Switching ---
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('demo-tab--active'));
      tab.classList.add('demo-tab--active');
      currentTab = tab.dataset.tab;

      const seqConfig = document.getElementById('sequence-config-section');

      if (currentTab === 'image') {
        if (taskTitle) taskTitle.textContent = 'Phân loại ảnh chuẩn (E1 / A1)';
        if (taskDesc) taskDesc.textContent = 'Tải ảnh lên hoặc chọn mẫu để đánh giá dự đoán từ Softmax, MLP, CNN hoặc Vision Transformer.';
        if (seqConfig) seqConfig.style.display = 'none';
      } else if (currentTab === 'sequence') {
        if (taskTitle) taskTitle.textContent = 'Mô hình hóa ảnh thành chuỗi (E3: LSTM / GRU)';
        if (taskDesc) taskDesc.textContent = 'Biến đổi ảnh thành chuỗi bước thời gian (quét từng hàng pixel hoặc dãy patch) và phân loại bằng RNN.';
        if (seqConfig) seqConfig.style.display = 'block';
      } else {
        if (taskTitle) taskTitle.textContent = 'Cơ chế chú ý Multi-Head Self-Attention (E2)';
        if (taskDesc) taskDesc.textContent = 'Đối chiếu giữa module Self-Attention tự viết từ đầu và PyTorch nn.MultiheadAttention trên các cách tokenize ảnh.';
        if (seqConfig) seqConfig.style.display = 'none';
      }

      updateModelOptions(currentTab);

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
      alert('Vui lòng chọn file hình ảnh hợp lệ (PNG, JPG, WEBP).');
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

  // --- 4. Prediction Execution (Simulated or Real API) ---
  if (predictBtn) {
    predictBtn.addEventListener('click', async () => {
      if (!currentFile && previewContainer.style.display === 'none') {
        alert('Vui lòng chọn hoặc tải lên một hình ảnh trước khi chạy suy luận!');
        return;
      }

      // UI Loading state
      predictBtn.disabled = true;
      predictBtn.textContent = 'Đang chạy suy luận...';
      const startTime = performance.now();

      try {
        // Simulated network latency (180ms - 350ms)
        await new Promise(resolve => setTimeout(resolve, 280));
        const endTime = performance.now();
        const latency = Math.round(endTime - startTime);

        // Generate mock predictions according to task
        let mockPredictions = [];
        if (currentTab === 'sequence') {
          mockPredictions = [
            { label: 'Chữ số 7 (Lớp 7)', confidence: 0.892 },
            { label: 'Chữ số 2 (Lớp 2)', confidence: 0.054 },
            { label: 'Chữ số 1 (Lớp 1)', confidence: 0.031 },
            { label: 'Chữ số 9 (Lớp 9)', confidence: 0.015 },
            { label: 'Chữ số 3 (Lớp 3)', confidence: 0.008 }
          ];
        } else if (currentTab === 'attention') {
          mockPredictions = [
            { label: 'Áo thun (T-shirt/Top)', confidence: 0.865 },
            { label: 'Áo sơ mi (Shirt)', confidence: 0.078 },
            { label: 'Áo khoác (Coat)', confidence: 0.032 },
            { label: 'Đầm (Dress)', confidence: 0.015 },
            { label: 'Áo len (Pullover)', confidence: 0.010 }
          ];
        } else {
          mockPredictions = [
            { label: 'Máy bay (Airplane)', confidence: 0.912 },
            { label: 'Chim (Bird)', confidence: 0.045 },
            { label: 'Tàu thủy (Ship)', confidence: 0.024 },
            { label: 'Ô tô (Automobile)', confidence: 0.011 },
            { label: 'Ngựa (Horse)', confidence: 0.008 }
          ];
        }

        renderResults(mockPredictions, latency);

      } catch (err) {
        console.error('Inference error:', err);
        alert('Có lỗi xảy ra trong quá trình suy luận. Xem console để biết chi tiết.');
      } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = '⚡ Chạy suy luận (Run Prediction)';
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
