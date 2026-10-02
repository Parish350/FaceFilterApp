import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const video = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');
let faceLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;
let lastVideoTime = -1;
let currentFilter = 'none';

// Assets
const spiderMaskImg = new Image();
spiderMaskImg.src = 'assets/spider_mask.jpg';
const spiderEyesImg = new Image();
spiderEyesImg.src = 'assets/spider_eyes.png';
const webOverlayImg = new Image();
webOverlayImg.src = 'assets/web_overlay.jpg';

// UI Elements
const splashScreen = document.getElementById('splash-screen');
const btnFlip = document.getElementById('btn-flip');
const btnCapture = document.getElementById('btn-capture');
const filterOptions = document.querySelectorAll('.filter-option');
const modePhoto = document.getElementById('mode-photo');
const modeVideo = document.getElementById('mode-video');
const flashEffect = document.getElementById('flash-effect');
const previewModal = document.getElementById('preview-modal');
const previewImage = document.getElementById('preview-image');
const previewVideo = document.getElementById('preview-video');
const btnClosePreview = document.getElementById('btn-close-preview');
const btnSavePreview = document.getElementById('btn-save-preview');

let facingMode = 'user';
let mediaRecorder;
let recordedChunks = [];
let isRecording = false;
let currentMode = 'PHOTO';

async function initializeFaceLandmarker() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            runningMode,
            numFaces: 1
        });
        
        // Hide splash screen when ready
        setTimeout(() => {
            splashScreen.style.opacity = '0';
            setTimeout(() => splashScreen.style.display = 'none', 500);
            enableCam();
        }, 1000);
    } catch (e) {
        console.error("Error initializing FaceLandmarker:", e);
        document.querySelector('.splash-text').innerText = "ERROR LOADING MODEL";
    }
}

async function enableCam() {
    if (!faceLandmarker) return;

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        video: { facingMode: facingMode, width: 1280, height: 720 }
    };

    try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
        webcamRunning = true;
    } catch (err) {
        console.error("Camera error:", err);
        alert("Camera access is required for this app to work.");
    }
}

btnFlip.addEventListener('click', () => {
    facingMode = facingMode === 'user' ? 'environment' : 'user';
    enableCam();
});

filterOptions.forEach(option => {
    option.addEventListener('click', () => {
        document.querySelector('.filter-option.active').classList.remove('active');
        option.classList.add('active');
        currentFilter = option.dataset.filter;
    });
});

async function predictWebcam() {
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;
    
    // Set recording resolution if needed, but we match canvas to video
    
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = faceLandmarker.detectForVideo(video, performance.now());
        
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // Draw video feed
        if (facingMode === 'user') {
            canvasCtx.save();
            canvasCtx.scale(-1, 1);
            canvasCtx.translate(-canvasElement.width, 0);
            canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
            canvasCtx.restore();
        } else {
            canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);
        }

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            drawFilters(results.faceLandmarks[0], results.faceBlendshapes ? results.faceBlendshapes[0] : null);
        }
    }

    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function drawFilters(landmarks, blendshapes) {
    // MediaPipe landmarks are normalized [0, 1]. Map to canvas coordinates
    const w = canvasElement.width;
    const h = canvasElement.height;
    
    // Flip x if user camera because canvas was flipped
    const getPoint = (idx) => {
        let x = landmarks[idx].x * w;
        if(facingMode === 'user') x = w - x;
        return { x, y: landmarks[idx].y * h };
    };

    // Common points
    const leftEye = getPoint(159);
    const rightEye = getPoint(386);
    const nose = getPoint(1);
    const chin = getPoint(152);
    const leftCheek = getPoint(234);
    const rightCheek = getPoint(454);
    const forehead = getPoint(10);
    
    const faceWidth = Math.hypot(rightCheek.x - leftCheek.x, rightCheek.y - leftCheek.y);
    const faceHeight = Math.hypot(chin.x - forehead.x, chin.y - forehead.y);
    const centerX = nose.x;
    const centerY = nose.y;
    const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

    // Face Outline (approximate jawline + forehead) for clipping
    const outlineIndices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
    
    if (currentFilter === 'spider_mask' || currentFilter === 'web_face') {
        canvasCtx.save();
        canvasCtx.beginPath();
        outlineIndices.forEach((idx, i) => {
            const p = getPoint(idx);
            if(i === 0) canvasCtx.moveTo(p.x, p.y);
            else canvasCtx.lineTo(p.x, p.y);
        });
        canvasCtx.closePath();
        canvasCtx.clip();
        
        canvasCtx.translate(centerX, centerY);
        canvasCtx.rotate(angle);
        
        const img = currentFilter === 'spider_mask' ? spiderMaskImg : webOverlayImg;
        if(img.complete) {
            canvasCtx.globalAlpha = currentFilter === 'spider_mask' ? 1.0 : 0.7;
            canvasCtx.globalCompositeOperation = currentFilter === 'spider_mask' ? 'source-over' : 'multiply';
            
            // Draw texture filling the face area
            canvasCtx.drawImage(img, -faceWidth*0.7, -faceHeight*0.7, faceWidth*1.4, faceHeight*1.4);
        }
        canvasCtx.restore();
    }
    
    if (currentFilter === 'spider_eyes' || currentFilter === 'spider_mask') {
        let blinkLeft = 0;
        let blinkRight = 0;
        
        if (blendshapes && blendshapes.categories) {
            const getScore = (name) => {
                const cat = blendshapes.categories.find(c => c.categoryName === name);
                return cat ? cat.score : 0;
            };
            blinkLeft = getScore('eyeBlinkLeft');
            blinkRight = getScore('eyeBlinkRight');
        }

        // Draw eyes
        canvasCtx.save();
        canvasCtx.translate(centerX, nose.y - faceHeight * 0.1);
        canvasCtx.rotate(angle);
        
        // Dimensions for the vector eyes (Bigger and thinner)
        const eyeWidth = faceWidth * 1.6; 
        const eyeHeight = eyeWidth * 0.45;
        
        // Visual left eye is actually the user's right eye if not mirrored, but we flipped the canvas.
        const vBlinkLeft = facingMode === 'user' ? blinkLeft : blinkRight;
        const vBlinkRight = facingMode === 'user' ? blinkRight : blinkLeft;
        
        // Scale Y to squint/close the eyes like a shutter (Negative to turn them upside down)
        const scaleYLeft = -Math.max(0.1, 1.0 - vBlinkLeft * 1.5);
        const scaleYRight = -Math.max(0.1, 1.0 - vBlinkRight * 1.5);
        
        const drawEyePath = (w, h) => {
            canvasCtx.beginPath();
            // Start at the inner corner (near the nose)
            canvasCtx.moveTo(w * 0.35, h * 0.1);
            // Top eyelid: relatively flat, slanted curve up to the outer top corner
            canvasCtx.quadraticCurveTo(-w * 0.1, -h * 0.3, -w * 0.45, -h * 0.4);
            // Bottom eyelid: deep, sweeping bowl curve back to the inner corner
            canvasCtx.quadraticCurveTo(-w * 0.2, h * 0.7, w * 0.35, h * 0.1);
            canvasCtx.closePath();
        };
        
        // Draw Visual Left Eye
        canvasCtx.save();
        canvasCtx.translate(-eyeWidth * 0.22, 0); 
        canvasCtx.scale(1, scaleYLeft);
        drawEyePath(eyeWidth * 0.4, eyeHeight);
        canvasCtx.fillStyle = '#fff';
        canvasCtx.fill();
        canvasCtx.lineWidth = eyeWidth * 0.04; // Thinner black border
        canvasCtx.strokeStyle = '#000';
        canvasCtx.lineJoin = 'round'; // Keeps the sharp corners slightly rounded and smooth
        canvasCtx.stroke();
        canvasCtx.restore();

        // Draw Visual Right Eye (Mirrored horizontally)
        canvasCtx.save();
        canvasCtx.translate(eyeWidth * 0.22, 0); 
        canvasCtx.scale(-1, scaleYRight); // scale X by -1 to mirror the exact same path
        drawEyePath(eyeWidth * 0.4, eyeHeight);
        canvasCtx.fillStyle = '#fff';
        canvasCtx.fill();
        canvasCtx.lineWidth = eyeWidth * 0.04; // Thinner black border
        canvasCtx.strokeStyle = '#000';
        canvasCtx.lineJoin = 'round';
        canvasCtx.stroke();
        canvasCtx.restore();

        canvasCtx.restore();
    }
}

// Modes & Capture
modePhoto.addEventListener('click', () => {
    currentMode = 'PHOTO';
    modePhoto.classList.add('active');
    modeVideo.classList.remove('active');
    btnCapture.classList.remove('recording');
});

modeVideo.addEventListener('click', () => {
    currentMode = 'VIDEO';
    modeVideo.classList.add('active');
    modePhoto.classList.remove('active');
});

btnCapture.addEventListener('click', () => {
    if (currentMode === 'PHOTO') {
        takePhoto();
    } else {
        toggleRecording();
    }
});

function takePhoto() {
    // Flash effect
    flashEffect.style.opacity = '1';
    setTimeout(() => flashEffect.style.opacity = '0', 100);
    
    const dataURL = canvasElement.toDataURL('image/jpeg', 0.9);
    showPreview(dataURL, 'photo');
}

function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

function startRecording() {
    recordedChunks = [];
    const stream = canvasElement.captureStream(30);
    const options = { mimeType: 'video/webm; codecs=vp9' };
    
    try {
        mediaRecorder = new MediaRecorder(stream, options);
    } catch (e) {
        mediaRecorder = new MediaRecorder(stream);
    }

    mediaRecorder.ondataavailable = handleDataAvailable;
    mediaRecorder.onstop = handleStop;
    mediaRecorder.start();
    isRecording = true;
    btnCapture.classList.add('recording');
}

function handleDataAvailable(event) {
    if (event.data.size > 0) {
        recordedChunks.push(event.data);
    }
}

function stopRecording() {
    mediaRecorder.stop();
    isRecording = false;
    btnCapture.classList.remove('recording');
}

function handleStop() {
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const videoURL = URL.createObjectURL(blob);
    showPreview(videoURL, 'video');
}

let currentPreviewData = null;
let currentPreviewType = null;

function showPreview(data, type) {
    currentPreviewData = data;
    currentPreviewType = type;
    previewModal.classList.remove('hidden');
    
    if (type === 'photo') {
        previewImage.src = data;
        previewImage.classList.remove('hidden');
        previewVideo.classList.add('hidden');
    } else {
        previewVideo.src = data;
        previewVideo.classList.remove('hidden');
        previewImage.classList.add('hidden');
    }
}

btnClosePreview.addEventListener('click', () => {
    previewModal.classList.add('hidden');
    previewVideo.pause();
    if(currentPreviewType === 'video') URL.revokeObjectURL(currentPreviewData);
});

btnSavePreview.addEventListener('click', () => {
    const a = document.createElement('a');
    a.href = currentPreviewData;
    a.download = currentPreviewType === 'photo' ? 'spider-hero.jpg' : 'spider-hero.webm';
    a.click();
});

// Initialize
initializeFaceLandmarker();
