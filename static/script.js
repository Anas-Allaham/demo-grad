let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const audioPlayback = document.getElementById("audioPlayback");
const reducedAudioPlayback = document.getElementById("reducedAudioPlayback");
const reducedAudioSection = document.getElementById("reducedAudioSection");
const reducedAudioLabel = document.getElementById("reducedAudioLabel");
const statusText = document.getElementById("statusText");
const toggleGuideBtn = document.getElementById("toggleGuideBtn");
const readTextBtn = document.getElementById("readTextBtn");
const stopReadBtn = document.getElementById("stopReadBtn");

const hasSpeechSynthesis = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
let availableVoices = [];
let activeUtterance = null;

if (hasSpeechSynthesis) {
    loadVoices();
    window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
} else if (readTextBtn && stopReadBtn) {
    readTextBtn.disabled = true;
    stopReadBtn.disabled = true;
    readTextBtn.title = "Text reader is not supported in this browser.";
}

if (toggleGuideBtn) {
    toggleGuideBtn.onclick = () => {
        const guide = document.getElementById("readingGuide");
        const hidden = guide.classList.toggle("hidden");
        toggleGuideBtn.textContent = hidden ? "Show Reader" : "Hide Reader";
    };
}

if (readTextBtn) {
    readTextBtn.onclick = () => {
        if (!hasSpeechSynthesis) {
            alert("Text reader is not supported in this browser.");
            return;
        }

        const text = document.getElementById("textInput").value.trim();
        if (!text) {
            alert("Please enter text first.");
            return;
        }

        stopTextReader(true);

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.voice = getPreferredVoice();
        utterance.rate = 0.92;
        utterance.pitch = 1;
        activeUtterance = utterance;

        utterance.onstart = () => {
            readTextBtn.disabled = true;
            stopReadBtn.disabled = false;
            statusText.textContent = "Reading text aloud...";
        };

        utterance.onend = () => {
            if (activeUtterance === utterance) {
                activeUtterance = null;
                readTextBtn.disabled = false;
                stopReadBtn.disabled = true;
                statusText.textContent = "Reader finished.";
            }
        };

        utterance.onerror = () => {
            if (activeUtterance === utterance) {
                activeUtterance = null;
                readTextBtn.disabled = false;
                stopReadBtn.disabled = true;
                statusText.textContent = "Reader failed.";
            }
        };

        window.speechSynthesis.speak(utterance);
    };
}

if (stopReadBtn) {
    stopReadBtn.onclick = () => {
        stopTextReader();
    };
}

startBtn.onclick = async () => {
    try {
        stopTextReader(true);
        audioChunks = [];
        recordedBlob = null;
        if (reducedAudioPlayback && reducedAudioSection) {
            reducedAudioPlayback.removeAttribute("src");
            reducedAudioPlayback.load();
            reducedAudioSection.classList.add("hidden");
        }

        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false,
            },
            video: false,
        });

        const track = stream.getAudioTracks()[0];
        if (track) {
            console.log("Actual microphone settings:", track.getSettings());
        }

        const preferredMimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
            ? "audio/webm;codecs=opus"
            : "";

        mediaRecorder = preferredMimeType
            ? new MediaRecorder(stream, { mimeType: preferredMimeType })
            : new MediaRecorder(stream);

        mediaRecorder.ondataavailable = event => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onerror = event => {
            console.error("MediaRecorder error:", event.error || event);
        };

        mediaRecorder.onstop = () => {
            const blobType = mediaRecorder.mimeType || "audio/webm";
            recordedBlob = new Blob(audioChunks, { type: blobType });
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
        if (mediaRecorder.state === "recording") {
            mediaRecorder.requestData();
        }
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
    formData.append("audio", recordedBlob, "raw_browser_recording.webm");

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
        let statusMessage = "Analysis complete.";
        if (data.reduced_audio_url && data.noise_reduction_applied) {
            statusMessage = "Analysis complete. You can now play the noise-reduced audio.";
        } else if (data.reduced_audio_url) {
            statusMessage = "Analysis complete. You can now play the processed audio.";
        } else if (!data.noise_reduction_applied) {
            statusMessage = "Analysis complete. Noise reduction package is not installed, so denoised playback is unavailable.";
        }

        const quality = data.audio_quality_check;
        if (quality && quality.possible_dropout) {
            statusMessage += " Warning: possible dropouts were detected in the raw recording.";
        }

        statusText.textContent = statusMessage;
    } catch (error) {
        document.getElementById("loading").classList.add("hidden");
        alert("Request failed: " + error.message);
        statusText.textContent = "Request failed.";
    }
};

function showResults(data) {
    document.getElementById("results").classList.remove("hidden");
    if (reducedAudioPlayback && reducedAudioSection) {
        if (data.reduced_audio_url) {
            reducedAudioPlayback.src = data.reduced_audio_url;
            if (reducedAudioLabel) {
                reducedAudioLabel.textContent = data.noise_reduction_applied ? "Noise-Reduced Playback" : "Processed Playback";
            }
            reducedAudioSection.classList.remove("hidden");
        } else {
            reducedAudioPlayback.removeAttribute("src");
            reducedAudioPlayback.load();
            reducedAudioSection.classList.add("hidden");
        }
    }

    document.getElementById("referenceIpa").textContent = data.reference_ipa;
    document.getElementById("predictedIpa").textContent = data.predicted_ipa;

    document.getElementById("correctCount").textContent = data.metrics.correct;
    document.getElementById("subCount").textContent = data.metrics.substitutions;
    document.getElementById("delCount").textContent = data.metrics.deletions;
    document.getElementById("insCount").textContent = data.metrics.insertions;
    document.getElementById("perValue").textContent = data.metrics.phoneme_error_rate + "%";

    showReadingGuide(data.reference_guide || []);
    const guide = document.getElementById("readingGuide");
    if (guide && toggleGuideBtn) {
        guide.classList.add("hidden");
        toggleGuideBtn.textContent = "Show Reader";
    }

    const table = document.getElementById("alignmentTable");
    table.innerHTML = "";

    data.alignment.forEach(row => {
        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>${escapeHtml(row.expected)}</td>
            <td>${escapeHtml(row.spoken)}</td>
            <td class="${row.result}">${escapeHtml(row.result)}</td>
            <td>${row.distance !== undefined ? escapeHtml(row.distance) : "—"}</td>
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


function showReadingGuide(guide) {
    const container = document.getElementById("readingGuide");
    if (!container) return;

    container.innerHTML = "";

    if (!guide.length) {
        container.innerHTML = `<p class="empty-guide">No phoneme guide available.</p>`;
        return;
    }

    guide.forEach(word => {
        const wordDiv = document.createElement("div");
        wordDiv.className = "guide-word";

        const title = document.createElement("h4");
        title.textContent = "Word " + word.word_index;
        wordDiv.appendChild(title);

        const table = document.createElement("table");
        table.innerHTML = `
            <thead>
                <tr>
                    <th>Phoneme</th>
                    <th>How to read it</th>
                    <th>Example</th>
                </tr>
            </thead>
            <tbody></tbody>
        `;

        const tbody = table.querySelector("tbody");
        word.phonemes.forEach(ph => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><strong>${escapeHtml(ph.symbol)}</strong></td>
                <td>${escapeHtml(ph.description)}</td>
                <td>${escapeHtml(ph.example)}</td>
            `;
            tbody.appendChild(row);
        });

        wordDiv.appendChild(table);
        container.appendChild(wordDiv);
    });
}

function loadVoices() {
    availableVoices = window.speechSynthesis.getVoices();
}

function getPreferredVoice() {
    if (!availableVoices.length) return null;

    const preferred = availableVoices.find(voice =>
        voice.lang && voice.lang.toLowerCase().startsWith("en")
    );

    return preferred || availableVoices[0];
}

function stopTextReader(silent = false) {
    if (!hasSpeechSynthesis) return;

    if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
        window.speechSynthesis.cancel();
    }

    activeUtterance = null;
    if (readTextBtn) readTextBtn.disabled = false;
    if (stopReadBtn) stopReadBtn.disabled = true;
    if (!silent) {
        statusText.textContent = "Reader stopped.";
    }
}
