// FORM SUBMIT
document.getElementById("visaForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const formData = new FormData(this);
    const data = Object.fromEntries(formData);

    const response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(data)
    });

    const result = await response.json();

    // Show result
    document.getElementById("popupStatus").innerHTML =
        result.status === "Approved"
            ? `<span class='success'>Visa Approved ✔</span>`
            : `<span class='fail'>Visa Denied ✖</span>`;

    document.getElementById("popupTime").innerHTML =
        `Estimated Processing Time: <b>${result.processing_time} days</b>`;

    window._whyText = result.why;

    document.getElementById("resultPopup").style.display = "flex";
});

// CLOSE POPUP
document.getElementById("closePopup").addEventListener("click", () => {
    document.getElementById("resultPopup").style.display = "none";
});

// WHY BUTTON
document.getElementById("whyBtn").addEventListener("click", () => {
    document.getElementById("whyPopup").style.display = "flex";
    document.getElementById("popupReason").innerText = window._whyText;
});

document.getElementById("closeWhy").addEventListener("click", () => {
    document.getElementById("whyPopup").style.display = "none";
});
