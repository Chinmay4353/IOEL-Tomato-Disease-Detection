const API_URL = '/api/latest';
const PREDICTIONS_URL = '/api/predictions';
const PREDICT_IMAGE_URL = '/api/predict-image';

const CONFIDENCE_THRESHOLD = 0.70;
const REFRESH_INTERVAL_MS = 5000;
const MAX_RECENT_PREDICTIONS = 8;

let selectedFile = null;
let cameraStream = null;
let selectedImageObjectUrl = null;

/* ============================================================
   DOM ELEMENTS
   ============================================================ */

const cameraButton = document.getElementById('camera-button');
const uploadButton = document.getElementById('upload-button');
const imageInput = document.getElementById('image-input');
const cameraPreview = document.getElementById('camera-preview');
const captureCanvas = document.getElementById('capture-canvas');
const captureButton = document.getElementById('capture-button');
const selectedImage = document.getElementById('selected-image');
const analyzeButton = document.getElementById('analyze-button');
const clearButton = document.getElementById('clear-button');
const analysisStatus = document.getElementById('analysis-status');
const analysisResult = document.getElementById('analysis-result');

const deviceStatus = document.getElementById('device-status');
const latestPrediction = document.getElementById('latest-prediction');
const latestConfidence = document.getElementById('latest-confidence');
const lastDetection = document.getElementById('last-detection');
const totalPredictions = document.getElementById('total-predictions');

const healthyCount = document.getElementById('tomato-count');
const diseaseCount = document.getElementById('not-tomato-count');
const averageConfidence = document.getElementById('average-confidence');

const predictionTableBody = document.getElementById(
    'prediction-table-body'
);

const latestImage = document.getElementById('latest-image');
const noImageMessage = document.getElementById(
    'no-image-message'
);

/* ============================================================
   INITIALIZATION
   ============================================================ */

function initializeDashboard() {
    bindEventListeners();
    resetDetectorState();
    fetchDashboardData();

    window.setInterval(
        fetchDashboardData,
        REFRESH_INTERVAL_MS
    );
}

function bindEventListeners() {
    cameraButton?.addEventListener('click', toggleCamera);

    uploadButton?.addEventListener('click', () => {
        imageInput?.click();
    });

    imageInput?.addEventListener(
        'change',
        handleFileSelection
    );

    captureButton?.addEventListener(
        'click',
        captureFrame
    );

    analyzeButton?.addEventListener(
        'click',
        analyzeSelectedImage
    );

    clearButton?.addEventListener(
        'click',
        clearSelectedImage
    );

    window.addEventListener('beforeunload', () => {
        stopCamera();
        revokeSelectedImageUrl();
    });
}

/* ============================================================
   GENERAL HELPERS
   ============================================================ */

function getConfidenceValue(confidence) {
    const value = Number(confidence);

    if (!Number.isFinite(value)) {
        return 0;
    }

    return Math.max(0, Math.min(1, value));
}

function formatConfidence(confidence) {
    return `${(getConfidenceValue(confidence) * 100).toFixed(1)}%`;
}

function formatPredictionLabel(prediction) {
    if (!prediction) {
        return 'Unknown';
    }

    return String(prediction)
        .replace(/_/g, ' ')
        .replace(/\s+/g, ' ')
        .trim()
        .replace(/\b\w/g, character => character.toUpperCase());
}

function getPredictionDisplay(prediction, confidence) {
    const confidenceValue = getConfidenceValue(confidence);
    const confidenceText = formatConfidence(confidenceValue);

    if (confidenceValue < CONFIDENCE_THRESHOLD) {
        return {
            label: 'Low Confidence / Uncertain',
            confidence: confidenceText,
            statusClass: 'result-uncertain',
            tableClass: 'prediction-status-uncertain',
            confidenceClass: 'prediction-confidence-uncertain',
            isLowConfidence: true
        };
    }

    const normalizedPrediction = String(prediction || '')
        .trim()
        .toLowerCase();

    if (normalizedPrediction === 'healthy') {
        return {
            label: 'Healthy',
            confidence: confidenceText,
            statusClass: 'result-healthy',
            tableClass: 'prediction-status-healthy',
            confidenceClass: 'prediction-confidence-healthy',
            isLowConfidence: false
        };
    }

    return {
        label: formatPredictionLabel(prediction),
        confidence: confidenceText,
        statusClass: 'result-disease',
        tableClass: 'prediction-status-disease',
        confidenceClass: 'prediction-confidence-disease',
        isLowConfidence: false
    };
}

function clearStatusClasses(element) {
    if (!element) {
        return;
    }

    element.classList.remove(
        'result-healthy',
        'result-disease',
        'result-uncertain'
    );
}

function setDeviceOnline(isOnline) {
    if (!deviceStatus) {
        return;
    }

    deviceStatus.textContent = isOnline
        ? 'ONLINE'
        : 'OFFLINE';

    deviceStatus.classList.toggle(
        'status-online',
        isOnline
    );

    deviceStatus.style.color = isOnline
        ? ''
        : '#f87171';
}

async function getJsonResponse(response, fallbackMessage) {
    let data = null;

    try {
        data = await response.json();
    } catch {
        throw new Error(fallbackMessage);
    }

    if (!response.ok) {
        throw new Error(
            data?.error ||
            data?.message ||
            fallbackMessage
        );
    }

    return data;
}

/* ============================================================
   CAMERA
   ============================================================ */

async function toggleCamera() {
    if (cameraStream) {
        closeCamera();
        return;
    }

    await openCamera();
}

async function openCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
        setAnalysisStatus(
            'Camera access is unavailable in this browser.'
        );
        return;
    }

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment'
            },
            audio: false
        });

        if (cameraPreview) {
            cameraPreview.srcObject = cameraStream;
            cameraPreview.style.display = 'block';
        }

        if (captureButton) {
            captureButton.hidden = false;
        }

        if (cameraButton) {
            cameraButton.textContent = 'Close Camera';
        }

        setAnalysisStatus(
            'Camera ready. Capture a frame to analyze it.'
        );
    } catch (error) {
        console.error('Camera error:', error);

        cameraStream = null;

        setAnalysisStatus(
            'Camera permission was denied or no camera was found.'
        );
    }
}

function closeCamera() {
    stopCamera();

    if (cameraPreview) {
        cameraPreview.srcObject = null;
        cameraPreview.style.display = 'none';
    }

    if (captureButton) {
        captureButton.hidden = true;
    }

    if (cameraButton) {
        cameraButton.textContent = 'Open Camera';
    }

    if (selectedFile) {
        setAnalysisStatus(
            `${selectedFile.name} is ready for analysis.`
        );
    } else {
        setAnalysisStatus(
            'Choose an image or open the camera.'
        );
    }
}

function stopCamera() {
    if (!cameraStream) {
        return;
    }

    cameraStream.getTracks().forEach(track => {
        track.stop();
    });

    cameraStream = null;
}

function captureFrame() {
    if (
        !cameraStream ||
        !cameraPreview ||
        !captureCanvas ||
        !cameraPreview.videoWidth ||
        !cameraPreview.videoHeight
    ) {
        setAnalysisStatus(
            'Camera is not ready yet. Please wait and try again.'
        );
        return;
    }

    captureCanvas.width = cameraPreview.videoWidth;
    captureCanvas.height = cameraPreview.videoHeight;

    const context = captureCanvas.getContext('2d');

    if (!context) {
        setAnalysisStatus(
            'Unable to capture the camera frame.'
        );
        return;
    }

    context.drawImage(
        cameraPreview,
        0,
        0,
        captureCanvas.width,
        captureCanvas.height
    );

    captureCanvas.toBlob(
        blob => {
            if (!blob) {
                setAnalysisStatus(
                    'Unable to create the captured image.'
                );
                return;
            }

            const file = new File(
                [blob],
                `camera_${Date.now()}.jpg`,
                {
                    type: 'image/jpeg'
                }
            );

            setSelectedFile(file);
        },
        'image/jpeg',
        0.90
    );
}

/* ============================================================
   FILE SELECTION
   ============================================================ */

function handleFileSelection(event) {
    const file = event.target?.files?.[0];

    if (!file) {
        return;
    }

    if (!file.type.startsWith('image/')) {
        setAnalysisStatus(
            'Please select a valid image file.'
        );

        if (imageInput) {
            imageInput.value = '';
        }

        return;
    }

    setSelectedFile(file);
}

function setSelectedFile(file) {
    if (!(file instanceof File)) {
        setAnalysisStatus(
            'The selected file is invalid.'
        );
        return;
    }

    revokeSelectedImageUrl();

    selectedFile = file;
    selectedImageObjectUrl = URL.createObjectURL(file);

    if (selectedImage) {
        selectedImage.src = selectedImageObjectUrl;
        selectedImage.style.display = 'block';
    }

    if (analyzeButton) {
        analyzeButton.disabled = false;
    }

    if (clearButton) {
        clearButton.hidden = false;
    }

    resetAnalysisResult();

    setAnalysisStatus(
        `${file.name} is ready for analysis.`
    );
}

function revokeSelectedImageUrl() {
    if (selectedImageObjectUrl) {
        URL.revokeObjectURL(selectedImageObjectUrl);
        selectedImageObjectUrl = null;
    }
}

function clearSelectedImage() {
    revokeSelectedImageUrl();

    selectedFile = null;

    if (imageInput) {
        imageInput.value = '';
    }

    if (selectedImage) {
        selectedImage.removeAttribute('src');
        selectedImage.style.display = 'none';
    }

    if (cameraStream) {
        closeCamera();
    } else if (cameraPreview) {
        cameraPreview.srcObject = null;
        cameraPreview.style.display = 'none';
    }

    if (captureButton) {
        captureButton.hidden = true;
    }

    if (clearButton) {
        clearButton.hidden = true;
    }

    if (analyzeButton) {
        analyzeButton.disabled = true;
    }

    resetAnalysisResult();

    setAnalysisStatus(
        'Choose an image or open the camera.'
    );
}

/* ============================================================
   ANALYSIS RESULT
   ============================================================ */

function resetAnalysisResult() {
    if (!analysisResult) {
        return;
    }

    analysisResult.hidden = true;
    analysisResult.textContent = '';
    clearStatusClasses(analysisResult);
}

function setAnalysisStatus(message) {
    if (analysisStatus) {
        analysisStatus.textContent = message;
    }
}

function showAnalysisResult(prediction, confidence) {
    if (!analysisResult) {
        return;
    }

    const display = getPredictionDisplay(
        prediction,
        confidence
    );

    clearStatusClasses(analysisResult);

    analysisResult.classList.add(
        display.statusClass
    );

    analysisResult.hidden = false;

    analysisResult.textContent =
        display.isLowConfidence
            ? `⚠ ${display.label} · ${display.confidence} confidence`
            : `${display.label} · ${display.confidence} confidence`;

    setAnalysisStatus(
        display.isLowConfidence
            ? 'Analysis complete. The model is uncertain about this image.'
            : 'Analysis complete.'
    );
}

/* ============================================================
   IMAGE ANALYSIS
   ============================================================ */

async function analyzeSelectedImage() {
    if (!selectedFile) {
        setAnalysisStatus(
            'Please select or capture an image first.'
        );
        return;
    }

    const formData = new FormData();

    formData.append('image', selectedFile);
    formData.append('device_id', 'dashboard-camera');

    if (analyzeButton) {
        analyzeButton.disabled = true;
    }

    setAnalysisStatus('Analyzing image...');
    resetAnalysisResult();

    try {
        const response = await fetch(
            PREDICT_IMAGE_URL,
            {
                method: 'POST',
                body: formData
            }
        );

        const result = await getJsonResponse(
            response,
            'Image analysis failed.'
        );

        const prediction =
            result?.label ??
            result?.prediction ??
            '';

        const confidence =
            getConfidenceValue(result?.confidence);

        showAnalysisResult(
            prediction,
            confidence
        );

        await fetchDashboardData();
    } catch (error) {
        console.error(
            'Image analysis error:',
            error
        );

        setAnalysisStatus(
            error instanceof Error
                ? error.message
                : 'Image analysis failed.'
        );
    } finally {
        if (analyzeButton) {
            analyzeButton.disabled = !selectedFile;
        }
    }
}

/* ============================================================
   DASHBOARD DATA
   ============================================================ */

async function fetchDashboardData() {
    try {
        const [
            latestResponse,
            predictionsResponse
        ] = await Promise.all([
            fetch(API_URL, {
                cache: 'no-store'
            }),
            fetch(PREDICTIONS_URL, {
                cache: 'no-store'
            })
        ]);

        const [latestData, predictionsData] =
            await Promise.all([
                getJsonResponse(
                    latestResponse,
                    'Unable to fetch latest prediction.'
                ),
                getJsonResponse(
                    predictionsResponse,
                    'Unable to fetch prediction history.'
                )
            ]);

        if (!Array.isArray(predictionsData)) {
            throw new Error(
                'Prediction history returned an invalid format.'
            );
        }

        setDeviceOnline(true);

        updateDashboard(
            latestData,
            predictionsData
        );
    } catch (error) {
        console.error(
            'Dashboard fetch error:',
            error
        );

        setDeviceOnline(false);
    }
}

/* ============================================================
   DASHBOARD UPDATE
   ============================================================ */

function updateDashboard(
    latestData,
    predictionsData
) {
    const predictions = Array.isArray(predictionsData)
        ? predictionsData
        : [];

    updateLatestPredictionCard(latestData);
    updateStatistics(predictions);
    updatePredictionTable(predictions);
    updateLatestImage(latestData);
}

function updateLatestPredictionCard(latestData) {
    const prediction =
        latestData?.prediction ||
        'No data';

    const hasConfidence =
        latestData?.confidence !== null &&
        latestData?.confidence !== undefined;

    const confidence = hasConfidence
        ? getConfidenceValue(latestData.confidence)
        : null;

    const display = confidence !== null
        ? getPredictionDisplay(
            prediction,
            confidence
        )
        : {
            label: formatPredictionLabel(prediction),
            confidence: '-',
            statusClass: 'result-uncertain'
        };

    if (latestPrediction) {
        latestPrediction.textContent = display.label;
        clearStatusClasses(latestPrediction);
        latestPrediction.classList.add(
            display.statusClass
        );
    }

    if (latestConfidence) {
        latestConfidence.textContent =
            display.confidence;

        clearStatusClasses(latestConfidence);
        latestConfidence.classList.add(
            display.statusClass
        );
    }

    if (lastDetection) {
        lastDetection.textContent =
            formatTimestamp(latestData?.timestamp);
    }

    if (totalPredictions) {
        /*
         * The API has already been verified to return the
         * complete prediction history. This is the authoritative
         * count shown by the dashboard.
         */
        // Set by updateStatistics to keep the data calculation
        // in one place.
    }
}

function updateStatistics(predictions) {
    const total = predictions.length;

    const confirmedHealthy = predictions.filter(item => {
        return (
            String(item?.prediction || '')
                .trim()
                .toLowerCase() === 'healthy' &&
            getConfidenceValue(item?.confidence) >=
                CONFIDENCE_THRESHOLD
        );
    }).length;

    const confirmedDisease = predictions.filter(item => {
        const prediction = String(
            item?.prediction || ''
        )
            .trim()
            .toLowerCase();

        return (
            prediction &&
            prediction !== 'healthy' &&
            getConfidenceValue(item?.confidence) >=
                CONFIDENCE_THRESHOLD
        );
    }).length;

    const average =
        total > 0
            ? (
                predictions.reduce(
                    (sum, item) => {
                        return (
                            sum +
                            getConfidenceValue(
                                item?.confidence
                            )
                        );
                    },
                    0
                ) / total
            ) * 100
            : 0;

    if (healthyCount) {
        healthyCount.textContent =
            String(confirmedHealthy);
    }

    if (diseaseCount) {
        diseaseCount.textContent =
            String(confirmedDisease);
    }

    if (averageConfidence) {
        averageConfidence.textContent =
            `${average.toFixed(1)}%`;
    }

    if (totalPredictions) {
        totalPredictions.textContent =
            String(total);
    }
}

/* ============================================================
   RECENT PREDICTIONS
   ============================================================ */

function updatePredictionTable(predictions) {
    if (!predictionTableBody) {
        return;
    }

    predictionTableBody.replaceChildren();

    /*
     * The backend has been verified to return newest-first.
     * We still sort defensively by timestamp and use id as a
     * fallback so the UI remains correct if API ordering changes.
     */
    const rows = predictions
        .slice()
        .sort(comparePredictions)
        .slice(0, MAX_RECENT_PREDICTIONS);

    if (rows.length === 0) {
        const row = document.createElement('tr');
        const cell = document.createElement('td');

        cell.colSpan = 4;
        cell.textContent = 'No predictions available.';
        cell.style.textAlign = 'center';

        row.appendChild(cell);
        predictionTableBody.appendChild(row);

        return;
    }

    rows.forEach(item => {
        predictionTableBody.appendChild(
            createPredictionRow(item)
        );
    });
}

function comparePredictions(a, b) {
    const timeA = Date.parse(a?.timestamp || '');
    const timeB = Date.parse(b?.timestamp || '');

    const validTimeA = Number.isFinite(timeA);
    const validTimeB = Number.isFinite(timeB);

    if (validTimeA && validTimeB && timeA !== timeB) {
        return timeB - timeA;
    }

    if (validTimeA !== validTimeB) {
        return validTimeA ? -1 : 1;
    }

    const idA = Number(a?.id);
    const idB = Number(b?.id);

    if (
        Number.isFinite(idA) &&
        Number.isFinite(idB)
    ) {
        return idB - idA;
    }

    return 0;
}

function createPredictionRow(item) {
    const row = document.createElement('tr');

    const prediction =
        item?.prediction ||
        'Unknown';

    const confidence =
        getConfidenceValue(item?.confidence);

    const display =
        getPredictionDisplay(
            prediction,
            confidence
        );

    const timeCell =
        document.createElement('td');

    const deviceCell =
        document.createElement('td');

    const predictionCell =
        document.createElement('td');

    const confidenceCell =
        document.createElement('td');

    timeCell.textContent =
        formatTimestamp(item?.timestamp);

    deviceCell.textContent =
        item?.device_id ||
        'N/A';

    predictionCell.textContent =
        display.isLowConfidence
            ? `⚠ ${display.label}`
            : display.label;

    confidenceCell.textContent =
        display.confidence;

    predictionCell.classList.add(
        display.tableClass
    );

    confidenceCell.classList.add(
        display.confidenceClass
    );

    row.append(
        timeCell,
        deviceCell,
        predictionCell,
        confidenceCell
    );

    return row;
}

function formatTimestamp(timestamp) {
    if (!timestamp) {
        return '-';
    }

    const date = new Date(timestamp);

    if (Number.isNaN(date.getTime())) {
        return '-';
    }

    return date.toLocaleString();
}

/* ============================================================
   LATEST CAPTURED IMAGE
   ============================================================ */

function updateLatestImage(latestData) {
    if (!latestImage) {
        return;
    }

    const imagePath =
        typeof latestData?.image_path === 'string'
            ? latestData.image_path.trim()
            : '';

    if (!imagePath) {
        hideLatestImage();
        return;
    }

    const imageUrl =
        `/captured-images/${encodeURIComponent(imagePath)}`;

    latestImage.onload = () => {
        latestImage.style.display = 'block';

        if (noImageMessage) {
            noImageMessage.style.display = 'none';
        }
    };

    latestImage.onerror = () => {
        hideLatestImage();
    };

    latestImage.src = imageUrl;
}

function hideLatestImage() {
    if (latestImage) {
        latestImage.removeAttribute('src');
        latestImage.style.display = 'none';
    }

    if (noImageMessage) {
        noImageMessage.style.display = 'block';
    }
}

/* ============================================================
   DETECTOR RESET
   ============================================================ */

function resetDetectorState() {
    if (analyzeButton) {
        analyzeButton.disabled = true;
    }

    if (clearButton) {
        clearButton.hidden = true;
    }

    if (captureButton) {
        captureButton.hidden = true;
    }

    if (cameraPreview) {
        cameraPreview.style.display = 'none';
    }

    if (selectedImage) {
        selectedImage.style.display = 'none';
    }

    resetAnalysisResult();

    setAnalysisStatus(
        'Choose an image or open the camera.'
    );
}

/* ============================================================
   START APPLICATION
   ============================================================ */

initializeDashboard();
