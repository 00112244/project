document.addEventListener('DOMContentLoaded', function() {
    // Index Page Navigation
    if (document.getElementById('detect-btn')) {
        const detectButton = document.getElementById('detect-btn');
        detectButton.addEventListener('click', function() {
            window.location.href = '/templates/detect.html';
        });
    }

    if (document.getElementById('custom-btn')) {
        const customButton = document.getElementById('custom-btn');
        customButton.addEventListener('click', function() {
            window.location.href = '/templates/custom.html';
        });
    }

    // Detect Page
    if (document.getElementById('start-btn')) {
        const startButton = document.getElementById('start-btn');
        const output = document.getElementById('output');
        const modelSelect = document.getElementById('model-select'); // Ensure this select exists

        startButton.addEventListener('click', function() {
            output.innerHTML = '<p>Detection started. Your camera is now active.</p>';

            // Start video stream
            const videoFeed = document.getElementById('video-feed'); // Ensure you have a video element with this ID
            videoFeed.src = `/video_feed/${modelSelect.value}`;
            videoFeed.play();

            // Here you can add logic for model predictions if needed
        });
    }

    // Custom Page
    if (document.getElementById('start-record-btn')) {
        const startRecordButton = document.getElementById('start-record-btn');
        const detectCustomButton = document.getElementById('detect-custom-btn');
        const output = document.getElementById('output');
        const customSignNameInput = document.getElementById('custom-sign-name'); // Ensure this input exists

        startRecordButton.addEventListener('click', function() {
            const signName = customSignNameInput.value.trim();
            if (!signName) {
                output.innerHTML = '<p>Please enter a name for your custom sign.</p>';
                return;
            }

            output.innerHTML = '<p>Recording started. Please show your custom sign.</p>';
            startRecordButton.disabled = true; // Disable the button to prevent repeated clicks

            // Simulating recording logic; replace this with actual recording logic
            setTimeout(() => {
                // AJAX call to save the recorded sign
                fetch('/record_custom_sign', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json', // Change to JSON
                    },
                    body: JSON.stringify({ sign_name: signName }), // Use JSON.stringify
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Network response was not ok');
                    }
                    return response.json(); // Return the JSON response
                })
                .then(data => {
                    output.innerHTML = `<p>${data.message}</p>`;
                })
                .catch(error => {
                    output.innerHTML = `<p>Error: ${error.message}</p>`;
                })
                .finally(() => {
                    startRecordButton.disabled = false; // Re-enable the button
                });
            }, 3000); // Simulate a delay in recording
        });

        detectCustomButton.addEventListener('click', function() {
            const signName = customSignNameInput.value.trim();
            if (!signName) {
                output.innerHTML = '<p>Please enter a name for your custom sign.</p>';
                return;
            }

            output.innerHTML = '<p>Detection started. Analyzing the recorded sign.</p>';

            // AJAX call to detect the custom sign
            fetch('/detect_custom_sign', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json', // Change to JSON
                },
                body: JSON.stringify({ sign_name: signName }), // Use JSON.stringify
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json(); // Return the JSON response
            })
            .then(data => {
                output.innerHTML = `<p>${data.message}</p>`;
            })
            .catch(error => {
                output.innerHTML = `<p>Error: ${error.message}</p>`;
            });
        });
    }
});
