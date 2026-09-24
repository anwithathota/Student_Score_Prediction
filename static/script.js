const form = document.getElementById("predictionForm");

const result = document.getElementById("result");

const predictedScore = document.getElementById("predictedScore");


form.addEventListener("submit", async function(event) {

    event.preventDefault();


    // Get values from the form

    const hoursStudied =
        parseFloat(document.getElementById("hours_studied").value);

    const previousScore =
        parseFloat(document.getElementById("previous_score").value);

    const attendance =
        parseFloat(document.getElementById("attendance").value);

    const sleepHours =
        parseFloat(document.getElementById("sleep_hours").value);


    // Send data to FastAPI

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({

                hours_studied: hoursStudied,

                previous_score: previousScore,

                attendance: attendance,

                sleep_hours: sleepHours

            })

        });


        // Get prediction from FastAPI

        const data = await response.json();


        // Display prediction

        predictedScore.textContent =
            data.predicted_score + " / 100";

        result.style.display = "block";


    } catch (error) {

        predictedScore.textContent =
            "Error connecting to server";

        result.style.display = "block";

        console.error(error);

    }

});