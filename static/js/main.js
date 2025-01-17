const form = document.getElementById('healthForm');


form.addEventListener('submit', function(event) {
    event.preventDefault();
    
    // Gather the form data
    const age = document.getElementById('age').value;
    const height = document.getElementById('height').value;
    const weight = document.getElementById('weight').value;
    const diabetic = document.getElementById('diabetic').checked? 1 : 0;
    const bloodpressure = document.getElementById('bloodPressure').checked? 1 : 0;
    const transplant = document.getElementById('transplant').checked? 1 : 0;
    const chronic = document.getElementById('chronicDisease').checked? 1 : 0;
    const allergy = document.getElementById('allergy').checked? 1 : 0;
    const cancer = document.getElementById('familyCancerHistory').checked? 1 : 0;
    const numberOfSurgeries = document.getElementById('numberOfSurgeries').value;


    // Create a JSON object to send in the request
    const requestData = {
        Age: parseInt(age),
        Diabetes: parseInt(diabetic),
        BloodPressureProblems: parseInt(bloodpressure),
        AnyTransplants: parseInt(transplant),
        AnyChronicDiseases: parseInt(chronic),
        Height: parseInt(height),
        Weight: parseInt(weight),
        KnownAllergies: parseInt(allergy),
        HistoryOfCancerInFamily: parseInt(cancer),
        NumberOfMajorSurgeries: parseInt(numberOfSurgeries)
    };

    // Send the request to the server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => response.json())
    .then(data => {
        message = "Estimated Premium: &#8377; " + parseFloat(data).toFixed(2);
        document.getElementById('predictionResult').innerHTML = message;
    })
    .catch(error => document.getElementById('predictionResult').innerText = error);
});