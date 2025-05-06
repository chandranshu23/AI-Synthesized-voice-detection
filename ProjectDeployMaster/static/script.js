// DOM Elements
const detectBtn = document.getElementById('detectBtn');
const fileInput = document.getElementById('fileInput');
const resultText = document.getElementById('resultText');
const loader = document.getElementById('loader');

// Function to simulate the voice detection API call
const detectVoice = async (file) => {
  // Show loader while detecting
  loader.style.display = 'block';
  resultText.textContent = '';

  try{
    const formData = new FormData();
    formData.append('file',file);

    const response = await fetch('/predict',{
      method: 'POST',
      body: formData
    });

    if(!response.ok){
      throw new Error('Prediction Failed');
    }

    const result = await response.json();

    loader.style.display = 'none';
    // resultText.textContent = `Detection Result: ${result}`;
    console.log(result.label);
    console.log(result.confidence);
    let displayConfidence;
    if(result.label ==="Human Voice"){
      displayConfidence = 100 - result.confidence;
    } else{
      displayConfidence = result.confidence * 100;
    }
    
    
    resultText.textContent = `Detection Result: ${result.label} (Confidence: ${(displayConfidence).toFixed(2)}%)`;

  } catch (error) {
    loader.style.display = 'none';
    resultText.textContent = 'Error: ' + error.message;
  }
};
//   // Simulate an API call with a delay (replace this with your actual API call)
//   setTimeout(() => {
//     // Hide loader after the "API call" is complete
//     loader.style.display = 'none';

//     // Simulate the detection result (replace this with actual API response logic)
//     const result = Math.random() > 0.5 ? 'AI-Generated' : 'Human Voice'; // Random result for demo purposes
//     resultText.textContent = `Detection Result: ${result}`;
//   }, 3000); // Simulating a 3-second delay for the "API call"
// };

// Event listener for Detect Button
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


function showFileName() {
  const fileInput = document.getElementById('fileInput');
  const display = document.getElementById('fileNameDisplay');
  const text = document.getElementById('fileNameText');

  if (fileInput.files.length > 0) {
    text.textContent = fileInput.files[0].name;
    display.style.display = 'flex';
  } else {
    display.style.display = 'none';
  }
}