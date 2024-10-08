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
