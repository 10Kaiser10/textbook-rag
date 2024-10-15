// Function to show the loading state on a button
function showLoading(button) {
    button.classList.add('loading');
    button.disabled = true;
}

// Function to hide the loading state on a button
function hideLoading(button) {
    button.classList.remove('loading');
    button.disabled = false;
}

function playWavFilesSequentially(wavFiles) {
    let currentIndex = 0;

    function playNext() {
        if (currentIndex < wavFiles.length) {
            // Create a new audio object for the current WAV file (base64 encoded)
            const audio = new Audio("data:audio/wav;base64," + wavFiles[currentIndex]);

            // Play the audio
            audio.play();

            // Once the current audio finishes, play the next one
            audio.onended = function() {
                currentIndex++;
                playNext(); // Play the next WAV file
            };
        }
    }

    // Start playing the first WAV file
    playNext();
}

// Function to play the speech using the TTS API
function playSpeechRAG() {
    const text = document.getElementById('question-response').innerText;

    fetch('https://textbook-rag.onrender.com/tts', {  // Replace with your actual TTS API URL
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({text})
    })
    .then(response => response.json())
    .then(data => {
        // Convert the received WAV string to a playable audio format
        playWavFilesSequentially(data.audios)
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Function to play the speech using the TTS API
function playSpeechAgent() {
    const text = document.getElementById('task-response').innerText;

    fetch('https://textbook-rag.onrender.com/tts', {  // Replace with your actual TTS API URL
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({text})
    })
    .then(response => response.json())
    .then(data => {
        // Convert the received WAV string to a playable audio format
        playWavFilesSequentially(data.audios)
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Function to submit the question to the chatbot API
function submitQuestion() {
    const query = document.getElementById('question-input').value;
    const submitButton = document.querySelector('button[onclick="submitQuestion()"]');

    // Show loading on the submit button
    showLoading(submitButton);

    // Call the API with the question
    fetch('https://textbook-rag.onrender.com/ask_rag', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query })
    })
    .then(response => response.json())
    .then(data => {
        // Display the returned string below the text input
        document.getElementById('question-response').innerText = data.answer;
        document.getElementById('speech-btn-rag').style.display = 'inline-block';
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('question-response').innerText = 'Error occurred';
    })
    .finally(() => {
        // Hide loading on the submit button
        hideLoading(submitButton);
    });
}

// Function to submit the task to another API
function submitTask() {
    const query = document.getElementById('task-input').value;
    const submitButton = document.querySelector('button[onclick="submitTask()"]');

    // Show loading on the submit button
    showLoading(submitButton);

    // Call the API with the task
    fetch('https://textbook-rag.onrender.com/ask_agent', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ query })
    })
    .then(response => response.json())
    .then(data => {
        // Display the first string below the first text box
        document.getElementById('task-response').innerText = data.answer;
        
        const taskImage = document.getElementById('task-image');
        // Load and display the image using the URL returned by the API
        if (data.imageUrl !== "") {
            taskImage.src = "../" + data.img_path;
            taskImage.style.display = 'block';
        } else {
            // If the image URL is empty, hide the image
            taskImage.style.display = 'none';
        }

        document.getElementById('speech-btn-agent').style.display = 'inline-block';
        // Display the float (if needed, add it somewhere else or display in the console)
        console.log('Float value:', data.img_scr);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('task-response').innerText = 'Error occurred';
    })
    .finally(() => {
        // Hide loading on the submit button
        hideLoading(submitButton);
    });
}
