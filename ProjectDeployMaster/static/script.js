// DOM Elements
const detectBtn = document.getElementById('detectBtn');
const fileInput = document.getElementById('fileInput');
const resultText = document.getElementById('resultText');
const loader = document.getElementById('loader');
const recordBtn = document.getElementById('recordBtn');
const recordStatus = document.getElementById('recordStatus');

// Function to simulate the voice detection API call
const detectVoice = async (file) => {
  loader.style.display = 'block';
  resultText.textContent = '';
  recordStatus.textContent = 'Recording complete. Detecting...'; // ✅ Add this here

  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) throw new Error('Prediction Failed');

    const result = await response.json();

    console.log("Raw response:", response);   // good for debug
    console.log("Parsed result:", result);

    loader.style.display = 'none';
    recordStatus.textContent = ''; // Clear after done

    let displayConfidence;
    if (result.label === "Human Voice") {
      displayConfidence = 100 - result.confidence;
    } else {
      displayConfidence = result.confidence * 100;
    }

    resultText.textContent = `Detection Result: ${result.label} (Confidence: ${displayConfidence.toFixed(2)}%)`;

  } catch (error) {
    loader.style.display = 'none';
    resultText.textContent = 'Error: ' + error.message;
    recordStatus.textContent = ''; // Clear in case of error
  }
};


// Event listener for Detect Button (file upload)
detectBtn.addEventListener('click', () => {
  // Check if a file has been selected
  if (!fileInput.files[0]) {
    alert('Please select an audio file to detect!');
    return;
  }

  const file = fileInput.files[0];

  // Call the detectVoice function with the selected file
  detectVoice(file);
});

// Function to show selected file name
function showFileName() {
  const fileInput = document.getElementById('fileInput');
  const display = document.getElementById('fileNameDisplay');
  const text = document.getElementById('fileNameText');
  const audioPlayer = document.getElementById('audio-playback');
  const file = fileInput.files[0];
  if (fileInput.files.length > 0) {
    text.textContent = fileInput.files[0].name;
    display.style.display = 'flex';
    const url = URL.createObjectURL(file);
    audioPlayer.src = url;
    audioPlayer.style.display = "block";
    audioPlayer.play();
  } else {
    display.style.display = 'none';
  }
}

// Record 12 seconds from mic and send to backend
async function recordAndDetect() {
  // ✅ Check for MediaRecorder support first
  if (typeof MediaRecorder === 'undefined') {
    alert("MediaRecorder is not supported in your browser.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
      audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
      const file = new File([audioBlob], 'recorded_audio.webm', { type: 'audio/webm' });
      const audioURL = URL.createObjectURL(audioBlob);
      // download file
      
      const a = document.createElement('a');
      a.href = audioURL;
      a.download = "recorded-audio.webm";
      a.textContent = "Download Recording";
      document.body.appendChild(a);
      document.body.removeChild(a);

      // audio playback
      const audioPlayer = document.getElementById('audio-playback');
      audioPlayer.src = audioURL;
      audioPlayer.style.display = "block";
      audioPlayer.play();




      // Convert WebM to WAV using ffmpeg.js in the browser
      // const wavFile = await convertWebMToWAV(file);
      // Send the converted WAV file to backend for detection
      detectVoice(file);
      // Reset UI
      recordBtn.disabled = false;
      recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Record 12s Audio';
      recordStatus.textContent = 'Recording complete. Detecting...';
    };

    // Disable record button during recording
    recordBtn.disabled = true;
    recordBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Recording...';
    recordStatus.textContent = 'Recording for 12 seconds...';

    mediaRecorder.start();
    console.log("Recording started...");
    
    // Stop recording after 12 seconds
    setTimeout(() => {
      if (mediaRecorder.state === "recording") {
        console.log("Stopping recording...");
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop()); // Stop the stream's tracks
      }
    }, 12000);
    
  } catch (err) {
    recordStatus.textContent = 'Microphone access failed: ' + err.message;
    recordBtn.disabled = false;
    recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Record 12s Audio';
  }
}


// Event listener for Record Button
recordBtn.addEventListener('click', recordAndDetect);

// Event listener for File Input (to clear status when a new file is selected)
document.getElementById('fileInput').addEventListener('change', function () {
  recordStatus.textContent = ''; // Clear status message like "Recording complete. Detecting..."
  recordBtn.disabled = false;
  recordBtn.innerHTML = '<i class="fas fa-microphone"></i> Record 12s Audio';
});
