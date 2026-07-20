document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('mlr-form');
  const resultContainer = document.getElementById('result-container');
  const finalMarkDisplay = document.getElementById('final-mark');

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    // Get input values
    const studyHours = parseFloat(document.getElementById('study-hours').value);
    const attendance = parseFloat(document.getElementById('attendance').value);
    const assignment = parseFloat(document.getElementById('assignment').value);

    // Validate inputs
    if (isNaN(studyHours) || isNaN(attendance) || isNaN(assignment)) {
      alert('Please enter valid numeric values for all fields.');
      return;
    }

    try {
      // Send request to Flask backend
      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          studyHours: studyHours,
          attendance: attendance,
          assignment: assignment
        })
      });

      const data = await response.json();

      if (data.status === 'success') {
        // Update UI
        resultContainer.classList.remove('show');
        
        // Add a slight delay for a nice animation effect
        setTimeout(() => {
          finalMarkDisplay.textContent = data.prediction.toFixed(1);
          resultContainer.classList.add('show');
        }, 100);
      } else {
        alert('Error: ' + data.error);
      }
    } catch (error) {
      console.error('Error predicting:', error);
      alert('Failed to connect to the server.');
    }
  });
});
