/* ================================
   ELEMENTS
================================ */
const fileInput = document.getElementById("fileInput");
const preview = document.getElementById("preview");
const form = document.getElementById("uploadForm");
const dropArea = document.getElementById("drop-area");

/* ================================
   FILE INPUT + VALIDATION + PREVIEW
================================ */
if (fileInput && preview) {
    fileInput.addEventListener("change", function () {
        const file = this.files[0];

        if (file) {
            const allowed = ["image/jpeg", "image/png", "image/jpg"];

            if (!allowed.includes(file.type)) {
                alert("❌ Only JPG/PNG images allowed!");
                fileInput.value = "";
                preview.style.display = "none";
                return;
            }

            preview.src = URL.createObjectURL(file);
            preview.style.display = "block";
        }
    });
}

/* ================================
   DRAG & DROP
================================ */
if (dropArea && fileInput && preview) {

    dropArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropArea.style.background = "rgba(255,255,255,0.2)";
    });

    dropArea.addEventListener("dragleave", () => {
        dropArea.style.background = "transparent";
    });

    dropArea.addEventListener("drop", (e) => {
        e.preventDefault();

        const files = e.dataTransfer.files;

        if (files.length > 0) {

            // FIX: Properly assign file
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(files[0]);
            fileInput.files = dataTransfer.files;

            preview.src = URL.createObjectURL(files[0]);
            preview.style.display = "block";
        }

        dropArea.style.background = "transparent";
    });
}

/* ================================
   SAMPLE IMAGE CLICK
================================ */
const sampleImages = document.querySelectorAll(".sample-card img");

sampleImages.forEach((img) => {
    img.addEventListener("click", function () {

        // Highlight selected
        sampleImages.forEach(i => i.style.border = "2px solid white");
        this.style.border = "3px solid #00e676";

        // Show preview
        if (preview) {
            preview.src = this.src;
            preview.style.display = "block";
        }

        // Send to backend
        fetch("/predict-sample", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image_path: this.getAttribute("src")
            })
        })
        .then(res => res.json())
        .then(data => showResult(data))
        .catch(err => console.log(err));
    });
});

/* ================================
   SHOW RESULT
================================ */
function showResult(data) {
    let resultBox = document.getElementById("resultBox");

    if (!resultBox) {
        resultBox = document.createElement("div");
        resultBox.id = "resultBox";
        resultBox.style.marginTop = "20px";
        resultBox.style.color = "white";
        document.querySelector(".hero").appendChild(resultBox);
    }

    resultBox.innerHTML = `
        <h2>${data.result}</h2>
        <h3>Confidence: ${data.confidence}%</h3>
    `;
}

/* ================================
   FORM SUBMIT LOADER
================================ */
if (form) {
    form.addEventListener("submit", function () {
        showLoader();
    });
}

function showLoader() {
    let loader = document.getElementById("loader");

    if (!loader) {
        loader = document.createElement("div");
        loader.id = "loader";
        loader.innerHTML = "🔍 Analyzing leaf...";
        loader.style.color = "white";
        loader.style.marginTop = "20px";

        document.querySelector(".hero").appendChild(loader);
    }
}

/* ================================
   OPTIONAL AUTO SUBMIT
================================ */
function autoSubmit() {
    if (fileInput && fileInput.files.length > 0 && form) {
        form.submit();
    }
}
