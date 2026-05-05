let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const audioPlayback = document.getElementById("audioPlayback");
const statusText = document.getElementById("statusText");

startBtn.onclick = async () => {
    try {
        audioChunks = [];
        recordedBlob = null;

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(audioChunks, { type: "audio/webm" });
            const audioUrl = URL.createObjectURL(recordedBlob);
            audioPlayback.src = audioUrl;
            analyzeBtn.disabled = false;
            statusText.textContent = "Recording saved. You can analyze now.";

            stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        analyzeBtn.disabled = true;
        statusText.textContent = "Recording...";
    } catch (error) {
        alert("Microphone access failed: " + error.message);
    }
};

stopBtn.onclick = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
};

analyzeBtn.onclick = async () => {
    const text = document.getElementById("textInput").value.trim();

    if (!text) {
        alert("Please enter text first.");
        return;
    }

    if (!recordedBlob) {
        alert("Please record audio first.");
        return;
    }

    const formData = new FormData();
    formData.append("text", text);
    formData.append("audio", recordedBlob, "recording.webm");

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results").classList.add("hidden");
    statusText.textContent = "Sending audio to local backend...";

    try {
        const response = await fetch("/analyze", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();
        document.getElementById("loading").classList.add("hidden");

        if (!response.ok || data.error) {
            alert(data.error || "Analysis failed.");
            statusText.textContent = "Analysis failed.";
            return;
        }

        showResults(data);
        statusText.textContent = "Analysis complete.";
    } catch (error) {
        document.getElementById("loading").classList.add("hidden");
        alert("Request failed: " + error.message);
        statusText.textContent = "Request failed.";
    }
};

function showResults(data) {
    document.getElementById("results").classList.remove("hidden");

    document.getElementById("referenceIpa").textContent = data.reference_ipa;
    document.getElementById("predictedIpa").textContent = data.predicted_ipa;

    document.getElementById("correctCount").textContent = data.metrics.correct;
    document.getElementById("subCount").textContent = data.metrics.substitutions;
    document.getElementById("delCount").textContent = data.metrics.deletions;
    document.getElementById("insCount").textContent = data.metrics.insertions;
    document.getElementById("perValue").textContent = data.metrics.phoneme_error_rate + "%";

    const table = document.getElementById("alignmentTable");
    table.innerHTML = "";

    data.alignment.forEach(row => {
        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>${escapeHtml(row.expected)}</td>
            <td>${escapeHtml(row.spoken)}</td>
            <td class="${row.result}">${escapeHtml(row.result)}</td>
        `;

        table.appendChild(tr);
    });
}

function escapeHtml(value) {
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}
