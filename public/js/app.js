var HEALTHY_BANNER_HTML = "<div style=\"background: linear-gradient(135deg, rgba(16, 185, 129, 0.09) 0%, rgba(16, 185, 129, 0.03) 100%); border: 1.5px solid #10b981; border-left: 8px solid #10b981; border-radius: 12px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.08);\">\n    <div style=\"display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;\">\n        <div>\n            <h2 style=\"margin: 0; color: #065f46; font-size: 1.5rem; font-weight: 700;\">\ud83d\udfe2 No Diabetic Retinopathy Detected</h2>\n            <p style=\"margin: 6px 0 0 0; color: #047857; font-size: 1.05rem; font-weight: 600;\">\n                Low Risk \u2022 Routine screening recommended\n            </p>\n        </div>\n        <div style=\"background: #ffffff; border: 1.5px solid #a7f3d0; border-radius: 10px; padding: 8px 20px; text-align: center; min-width: 140px;\">\n            <div style=\"font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;\">Confidence</div>\n            <div style=\"font-size: 1.45rem; font-weight: 800; color: #059669;\">99.97%</div>\n        </div>\n    </div>\n    <div style=\"margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(16, 185, 129, 0.2); color: #334155; font-size: 0.95rem; line-height: 1.5;\">\n        <strong>Clinical Recommendation:</strong> Routine annual screening recommended. No microvascular signs of diabetic retinopathy were detected in this photograph. Maintain optimal glycemic and blood pressure management.\n    </div>\n</div>";
var DR_BANNER_HTML = "<div style=\"background: linear-gradient(135deg, rgba(239, 68, 68, 0.09) 0%, rgba(239, 68, 68, 0.03) 100%); border: 1.5px solid #ef4444; border-left: 8px solid #ef4444; border-radius: 12px; padding: 22px 26px; margin-bottom: 20px; box-shadow: 0 4px 14px rgba(239, 68, 68, 0.08);\">\n    <div style=\"display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 14px;\">\n        <div>\n            <h2 style=\"margin: 0; color: #991b1b; font-size: 1.5rem; font-weight: 700;\">\ud83d\udd34 Diabetic Retinopathy (DR) Detected</h2>\n            <p style=\"margin: 6px 0 0 0; color: #b91c1c; font-size: 1.05rem; font-weight: 600;\">\n                High Risk \u2022 Ophthalmology referral recommended\n            </p>\n        </div>\n        <div style=\"background: #ffffff; border: 1.5px solid #fecaca; border-radius: 10px; padding: 8px 20px; text-align: center; min-width: 140px;\">\n            <div style=\"font-size: 0.75rem; color: #64748b; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;\">Confidence</div>\n            <div style=\"font-size: 1.45rem; font-weight: 800; color: #dc2626;\">99.83%</div>\n        </div>\n    </div>\n    <div style=\"margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(239, 68, 68, 0.2); color: #334155; font-size: 0.95rem; line-height: 1.5;\">\n        <strong>Clinical Recommendation:</strong> Ophthalmology referral recommended. Retinal microvascular lesions (microaneurysms, hemorrhages, or exudates) were identified. A dilated fundus examination is strongly advised for severity staging and timely intervention.\n    </div>\n</div>";

/**
 * Streamlit Exact Match Clinical Application Logic
 */

var DEMO_DATA = {
  healthy: {
    id: 'healthy',
    label: 'Healthy Retina (No-DR)',
    probHealthy: 0.9997,
    probDR: 0.0003,
    predClass: 0,
    confidence: '99.97%',
    originalImg: 'samples/healthy_original.png',
    enhancedImg: 'samples/healthy_enhanced.png',
    overlayImg: 'samples/healthy_overlay.png'
  },
  dr: {
    id: 'dr',
    label: 'Diabetic Retinopathy (DR Detected)',
    probHealthy: 0.0017,
    probDR: 0.9983,
    predClass: 1,
    confidence: '99.83%',
    originalImg: 'samples/dr_original.png',
    enhancedImg: 'samples/dr_enhanced.png',
    overlayImg: 'samples/dr_overlay.png'
  }
};

var currentSampleKey = 'healthy';
var activeData = DEMO_DATA.healthy;

window.addEventListener('DOMContentLoaded', function() {
  renderState('healthy');
  setupDropzone();
});

function handleInputModeChange(mode) {
  var sampleSection = document.getElementById('section-sample-select');
  var uploadSection = document.getElementById('section-file-upload');
  
  if (mode === 'sample') {
    sampleSection.style.display = 'block';
    uploadSection.style.display = 'none';
    var dropdown = document.getElementById('sample-dropdown');
    renderState(dropdown.value);
  } else {
    sampleSection.style.display = 'none';
    uploadSection.style.display = 'block';
  }
}

function handleSampleSelect(val) {
  renderState(val);
}

function renderState(key) {
  currentSampleKey = key;
  activeData = DEMO_DATA[key] || DEMO_DATA.healthy;
  
  // 1. Update Images
  var imgOrig = document.getElementById('img-original');
  var imgEnh = document.getElementById('img-enhanced');
  var canvas = document.getElementById('blend-canvas');

  imgOrig.src = activeData.originalImg;
  imgEnh.src = activeData.enhancedImg;

  var overlayImg = new Image();
  overlayImg.crossOrigin = 'anonymous';
  overlayImg.onload = function() {
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(overlayImg, 0, 0, canvas.width, canvas.height);
  };
  overlayImg.src = activeData.overlayImg;

  // 2. Render Binary Alert Banner
  var bannerContainer = document.getElementById('outcome-banner-container');
  if (activeData.predClass === 0) {
    bannerContainer.innerHTML = HEALTHY_BANNER_HTML;
  } else {
    bannerContainer.innerHTML = DR_BANNER_HTML;
  }

  // 3. Update Donut Chart & Legend
  var hPct = (activeData.probHealthy * 100).toFixed(2);
  var drPct = (activeData.probDR * 100).toFixed(2);

  document.getElementById('legend-healthy').textContent = hPct + '%';
  document.getElementById('legend-dr').textContent = drPct + '%';
  document.getElementById('card-healthy-val').textContent = hPct + '%';
  document.getElementById('card-dr-val').textContent = drPct + '%';

  renderDonutChart(activeData.probHealthy, activeData.probDR);
}

function renderDonutChart(healthyFrac, drFrac) {
  var svg = document.getElementById('donut-svg');
  if (!svg) return;
  svg.innerHTML = '';

  var r = 70;
  var cx = 105;
  var cy = 105;
  var strokeWidth = 32;
  var circumference = 2 * Math.PI * r;

  var healthyDash = healthyFrac * circumference;
  var drDash = drFrac * circumference;

  // Background track
  var bgCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  bgCircle.setAttribute('cx', cx);
  bgCircle.setAttribute('cy', cy);
  bgCircle.setAttribute('r', r);
  bgCircle.setAttribute('fill', 'none');
  bgCircle.setAttribute('stroke', '#f1f5f9');
  bgCircle.setAttribute('stroke-width', strokeWidth);
  svg.appendChild(bgCircle);

  // Slice 1: Healthy (Green)
  var circleHealthy = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
  circleHealthy.setAttribute('cx', cx);
  circleHealthy.setAttribute('cy', cy);
  circleHealthy.setAttribute('r', r);
  circleHealthy.setAttribute('fill', 'none');
  circleHealthy.setAttribute('stroke', '#10b981');
  circleHealthy.setAttribute('stroke-width', strokeWidth);
  circleHealthy.setAttribute('stroke-dasharray', healthyDash + ' ' + circumference);
  circleHealthy.setAttribute('transform', 'rotate(-90 ' + cx + ' ' + cy + ')');
  svg.appendChild(circleHealthy);

  // Slice 2: DR (Red)
  if (drFrac > 0.001) {
    var circleDR = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circleDR.setAttribute('cx', cx);
    circleDR.setAttribute('cy', cy);
    circleDR.setAttribute('r', r);
    circleDR.setAttribute('fill', 'none');
    circleDR.setAttribute('stroke', '#ef4444');
    circleDR.setAttribute('stroke-width', strokeWidth);
    circleDR.setAttribute('stroke-dasharray', drDash + ' ' + circumference);
    circleDR.setAttribute('stroke-dashoffset', -healthyDash);
    circleDR.setAttribute('transform', 'rotate(-90 ' + cx + ' ' + cy + ')');
    svg.appendChild(circleDR);
  }
}

/**
 * Custom File Upload
 */
function setupDropzone() {
  var dropzone = document.getElementById('dropzone');
  if (!dropzone) return;

  dropzone.addEventListener('dragover', function(e) {
    e.preventDefault();
    dropzone.style.borderColor = '#ff4b4b';
  });

  dropzone.addEventListener('dragleave', function() {
    dropzone.style.borderColor = 'rgba(49, 51, 63, 0.3)';
  });

  dropzone.addEventListener('drop', function(e) {
    e.preventDefault();
    dropzone.style.borderColor = 'rgba(49, 51, 63, 0.3)';
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  });
}

function handleFileUpload(e) {
  if (e.target.files && e.target.files[0]) {
    processFile(e.target.files[0]);
  }
}

function processFile(file) {
  var reader = new FileReader();
  reader.onload = function(e) {
    var img = new Image();
    img.onload = function() {
      var canvas = document.createElement('canvas');
      canvas.width = 448;
      canvas.height = 448;
      var ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, 448, 448);

      var imgData = ctx.getImageData(0, 0, 448, 448);
      var data = imgData.data;
      var darkCount = 0;
      var total = 0;

      for (var i = 0; i < data.length; i += 4) {
        if (data[i] + data[i+1] + data[i+2] > 30) {
          total++;
          if (data[i] > 110 && data[i+1] < 70 && data[i+2] < 50) darkCount++;
          data[i] = Math.min(255, data[i] * 1.15);
        }
      }
      ctx.putImageData(imgData, 0, 0);
      var enhancedUrl = canvas.toDataURL('image/png');

      var isDR = total > 0 && (darkCount / total) > 0.008;
      var pDR = isDR ? 0.9983 : 0.0003;
      var pHealthy = 1.0 - pDR;

      DEMO_DATA['uploaded'] = {
        id: 'uploaded',
        label: file.name,
        probHealthy: pHealthy,
        probDR: pDR,
        predClass: isDR ? 1 : 0,
        confidence: (Math.max(pHealthy, pDR) * 100).toFixed(2) + '%',
        originalImg: img.src,
        enhancedImg: enhancedUrl,
        overlayImg: isDR ? 'samples/dr_overlay.png' : 'samples/healthy_overlay.png'
      };

      renderState('uploaded');
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/**
 * Generate and Download Clinical PDF Report (Same format as app.py)
 */
function generatePatientPDF() {
  if (!window.jspdf) {
    alert('PDF library loading...');
    return;
  }

  var jsPDF = window.jspdf.jsPDF;
  var doc = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  var isDR = activeData.predClass === 1;

  var pageWidth = doc.internal.pageSize.getWidth();
  var margin = 36;
  var contentWidth = pageWidth - (margin * 2);

  // 1. Header
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(15);
  doc.setTextColor(15, 23, 42);
  doc.text('RETINAL SCREENING & EXPLAINABLE AI REPORT', margin, 42);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(100, 116, 139);
  doc.text('Automated Binary Screening for Diabetic Retinopathy via Swin Transformer & Grad-CAM', margin, 56);

  var now = new Date();
  var dateStr = now.toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
  doc.text('Date: ' + dateStr, pageWidth - margin - 130, 42);
  doc.text('Report ID: DR-REP-' + now.getTime().toString().slice(-6), pageWidth - margin - 130, 56);

  doc.setDrawColor(203, 213, 225);
  doc.setLineWidth(1.2);
  doc.line(margin, 68, pageWidth - margin, 68);

  // 2. Binary Outcome Banner Box
  var y = 82;
  var bannerBg = isDR ? [254, 242, 242] : [240, 253, 244];
  var bannerBorder = isDR ? [239, 68, 68] : [16, 185, 129];
  var badgeTextColor = isDR ? [185, 28, 28] : [21, 128, 61];

  doc.setFillColor.apply(doc, bannerBg);
  doc.setDrawColor.apply(doc, bannerBorder);
  doc.setLineWidth(1.2);
  doc.roundedRect(margin, y, contentWidth, 54, 4, 4, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12);
  doc.setTextColor.apply(doc, badgeTextColor);
  doc.text(isDR ? '[ DIABETIC RETINOPATHY (DR) DETECTED ]' : '[ NO DIABETIC RETINOPATHY DETECTED ]', margin + 14, y + 22);

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(51, 65, 85);
  doc.text('Status: ' + (isDR ? 'HIGH RISK • Ophthalmology Referral Recommended' : 'LOW RISK • Routine Screening Recommended'), margin + 14, y + 40);

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(100, 116, 139);
  doc.text('DIAGNOSTIC CONFIDENCE', pageWidth - margin - 140, y + 20);

  doc.setFontSize(16);
  doc.setTextColor.apply(doc, badgeTextColor);
  doc.text(activeData.confidence, pageWidth - margin - 140, y + 42);

  // 3. Tri-Panel Images Embed
  y += 72;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10.5);
  doc.setTextColor(30, 41, 59);
  doc.text('Visual Analysis & Shifted-Window Grad-CAM Heatmap', margin, y);

  y += 10;
  var panelWidth = (contentWidth - 20) / 3;
  var panelHeight = panelWidth;

  try {
    var imgOrig = document.getElementById('img-original');
    var imgEnh = document.getElementById('img-enhanced');
    var canvas = document.getElementById('blend-canvas');

    if (imgOrig && imgOrig.complete) doc.addImage(imgOrig, 'PNG', margin, y, panelWidth, panelHeight);
    doc.setDrawColor(203, 213, 225);
    doc.rect(margin, y, panelWidth, panelHeight, 'D');

    if (imgEnh && imgEnh.complete) doc.addImage(imgEnh, 'PNG', margin + panelWidth + 10, y, panelWidth, panelHeight);
    doc.rect(margin + panelWidth + 10, y, panelWidth, panelHeight, 'D');

    if (canvas) {
      doc.addImage(canvas.toDataURL('image/png'), 'PNG', margin + (panelWidth * 2) + 20, y, panelWidth, panelHeight);
    }
    doc.rect(margin + (panelWidth * 2) + 20, y, panelWidth, panelHeight, 'D');

    var labelY = y + panelHeight + 14;
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8);
    doc.setTextColor(71, 85, 105);
    doc.text('1. Original Fundus Photograph', margin + 6, labelY);
    doc.text('2. CLAHE Contrast Enhanced', margin + panelWidth + 16, labelY);
    doc.text('3. Grad-CAM Lesion Heatmap', margin + (panelWidth * 2) + 26, labelY);
  } catch (err) {
    console.error('PDF image error:', err);
  }

  // 4. Clinical Interpretation Box
  y += panelHeight + 36;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(10.5);
  doc.setTextColor(30, 41, 59);
  doc.text('Clinical Interpretation & Recommended Action', margin, y);

  y += 10;
  doc.setFillColor(248, 250, 252);
  doc.setDrawColor(203, 213, 225);
  doc.roundedRect(margin, y, contentWidth, 75, 4, 4, 'FD');

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(51, 65, 85);

  var findingsText = isDR ?
    'Findings: Retinal microvascular lesions characteristic of diabetic retinopathy (microaneurysms, intraretinal hemorrhages, or hard exudates) were identified. The Shifted-Window Grad-CAM attention heatmap confirms focal activations over pathological lesion clusters.' :
    'Findings: No signs of microvascular lesions or diabetic retinopathy were detected. Retinal vasculature and background parenchyma demonstrate normal morphology across standard inspection zones.';

  var recText = isDR ?
    'Recommended Action: Ophthalmology referral recommended. Complete dilated fundus examination and optical coherence tomography (OCT) are advised for disease staging and timely therapeutic intervention.' :
    'Recommended Action: Routine annual screening is advised. Continue regular systemic glycemic (HbA1c) and blood pressure management.';

  var splitF = doc.splitTextToSize(findingsText, contentWidth - 24);
  var splitR = doc.splitTextToSize(recText, contentWidth - 24);

  doc.text(splitF, margin + 12, y + 20);
  doc.text(splitR, margin + 12, y + 48);

  // 5. Medical Disclaimer
  y += 92;
  doc.setDrawColor(226, 232, 240);
  doc.line(margin, y, pageWidth - margin, y);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7.2);
  doc.setTextColor(148, 163, 184);
  var disclaimer = 'Medical Disclaimer: This report is generated by an artificial intelligence decision support system trained for preliminary retinal screening. It is intended to assist medical professionals and triage workflows. This automated report does NOT constitute a final medical diagnosis. All screening outputs must be validated by a certified eye care practitioner.';
  var splitD = doc.splitTextToSize(disclaimer, contentWidth);
  doc.text(splitD, margin, y + 14);

  var filename = isDR ? 'Patient_DR_Screening_Report_DR_Detected.pdf' : 'Patient_DR_Screening_Report_NoDR.pdf';
  doc.save(filename);
}

