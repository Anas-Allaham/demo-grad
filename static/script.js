let mediaRecorder;
let audioChunks = [];
let recordedBlob = null;
let originalAudioUrl = null;

let activeProfile = localStorage.getItem("pronunciation_profile") || "";
let currentSentenceId = null;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const audioPlayback = document.getElementById("audioPlayback");
const reducedAudioPlayback = document.getElementById("reducedAudioPlayback");
const reducedAudioSection = document.getElementById("reducedAudioSection");
const reducedAudioLabel = document.getElementById("reducedAudioLabel");
const playBeforeBtn = document.getElementById("playBeforeBtn");
const playAfterBtn = document.getElementById("playAfterBtn");
const statusText = document.getElementById("statusText");
const toggleGuideBtn = document.getElementById("toggleGuideBtn");
const readTextBtn = document.getElementById("readTextBtn");
const stopReadBtn = document.getElementById("stopReadBtn");

const hasSpeechSynthesis = "speechSynthesis" in window && "SpeechSynthesisUtterance" in window;
let availableVoices = [];
let activeUtterance = null;

configurePlaybackButton(
    playBeforeBtn,
    audioPlayback,
    "Listen Before Edits",
    "Pause Before Edits",
    reducedAudioPlayback,
);
configurePlaybackButton(
    playAfterBtn,
    reducedAudioPlayback,
    "Listen After Edits",
    "Pause After Edits",
    audioPlayback,
);

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
        if (originalAudioUrl) {
            URL.revokeObjectURL(originalAudioUrl);
            originalAudioUrl = null;
        }
        audioPlayback.removeAttribute("src");
        audioPlayback.load();
        playBeforeBtn.disabled = true;
        playAfterBtn.disabled = true;
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
            originalAudioUrl = URL.createObjectURL(recordedBlob);
            audioPlayback.src = originalAudioUrl;
            playBeforeBtn.disabled = false;
            analyzeBtn.disabled = false;
            statusText.textContent = "Recording saved. Listen before edits, then analyze it.";

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
    formData.append("include_processed_audio", "1");
    if (activeProfile) {
        formData.append("user", activeProfile);
    }
    if (currentSentenceId !== null) {
        formData.append("sentence_id", currentSentenceId);
    }

    document.getElementById("loading").classList.remove("hidden");
    document.getElementById("results").classList.add("hidden");
    statusText.textContent = "Cleaning and analyzing audio... This can take about 30 seconds.";

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

        // Audio-quality gate: an unscorable recording is not scored and does
        // not update mastery. Ask the user to record again.
        if (data.scorable === false) {
            document.getElementById("results").classList.add("hidden");
            const reasons = (data.audio_quality && data.audio_quality.reasons || []).join(", ");
            statusText.textContent =
                (data.message || "That recording could not be scored. Please record again.") +
                (reasons ? " (" + reasons + ")" : "");
            currentSentenceId = null;
            return;
        }

        showResults(data);
        let statusMessage = "Analysis complete.";
        if (data.scoring_trusted === false && data.mastery_note) {
            statusMessage = data.mastery_note;
        }
        if (data.cleanvoice_applied) {
            statusMessage = "Analysis complete. Cleanvoice enhanced the recording before scoring.";
        } else if (data.cleanvoice_error) {
            statusMessage = "Analysis complete. Cleanvoice was unavailable, so local audio cleanup was used.";
        } else if ((data.processed_audio_data_url || data.reduced_audio_url) && data.noise_reduction_applied) {
            statusMessage = "Analysis complete. You can now play the noise-reduced audio.";
        } else if (data.processed_audio_data_url || data.reduced_audio_url) {
            statusMessage = "Analysis complete. You can now play the processed audio.";
        } else if (!data.noise_reduction_applied) {
            statusMessage = "Analysis complete. Noise reduction package is not installed, so denoised playback is unavailable.";
        }

        const quality = data.audio_quality;
        if (quality && quality.metrics && quality.quality_weight !== undefined && quality.quality_weight < 0.75) {
            statusMessage += " Note: audio quality was borderline.";
        }

        if (data.profile && data.mastery_updated) {
            statusMessage += ` Counted toward ${data.profile.name}'s progress.`;
        } else if (data.profile && data.scoring_trusted === false) {
            statusMessage += " (Provisional score shown; progress not updated.)";
        }

        statusText.textContent = statusMessage;

        // The exercise this attempt was scored against is now used up --
        // require fetching a fresh one before the assignment link is reused.
        currentSentenceId = null;
    } catch (error) {
        document.getElementById("loading").classList.add("hidden");
        alert("Request failed: " + error.message);
        statusText.textContent = "Request failed.";
    }
};

function showResults(data) {
    document.getElementById("results").classList.remove("hidden");
    if (reducedAudioPlayback && reducedAudioSection) {
        const processedAudioSource = data.processed_audio_data_url || data.reduced_audio_url;
        if (processedAudioSource) {
            reducedAudioPlayback.src = processedAudioSource;
            playAfterBtn.disabled = false;
            if (reducedAudioLabel) {
                reducedAudioLabel.textContent = data.cleanvoice_applied
                    ? "Cleanvoice-Enhanced Playback"
                    : (data.noise_reduction_applied ? "Noise-Reduced Playback" : "Processed Playback");
            }
            reducedAudioSection.classList.remove("hidden");
        } else {
            reducedAudioPlayback.removeAttribute("src");
            reducedAudioPlayback.load();
            playAfterBtn.disabled = true;
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

function configurePlaybackButton(button, player, playLabel, pauseLabel, otherPlayer) {
    if (!button || !player) {
        return;
    }

    button.onclick = async () => {
        if (!player.hasAttribute("src")) {
            return;
        }
        if (player.paused) {
            if (otherPlayer) {
                otherPlayer.pause();
            }
            try {
                await player.play();
            } catch (error) {
                statusText.textContent = "Audio playback failed: " + error.message;
            }
        } else {
            player.pause();
        }
    };

    player.addEventListener("play", () => {
        button.textContent = pauseLabel;
    });
    const showPlayLabel = () => {
        button.textContent = playLabel;
    };
    player.addEventListener("pause", showPlayLabel);
    player.addEventListener("ended", showPlayLabel);
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

// -----------------------------
// Profiles -- no auth, a "profile" is just a name (see db.get_or_create_user)
// -----------------------------
const profileSelect = document.getElementById("profileSelect");
const profileStatus = document.getElementById("profileStatus");
const NEW_PROFILE_VALUE = "__new__";

async function loadProfiles(selectAfterLoad) {
    try {
        const response = await fetch("/users");
        const users = await response.json();

        profileSelect.innerHTML = "";
        const noneOption = document.createElement("option");
        noneOption.value = "";
        noneOption.textContent = "No profile (Free Practice only)";
        profileSelect.appendChild(noneOption);

        users.forEach(user => {
            const option = document.createElement("option");
            option.value = user.name;
            option.textContent = user.name;
            profileSelect.appendChild(option);
        });

        const newOption = document.createElement("option");
        newOption.value = NEW_PROFILE_VALUE;
        newOption.textContent = "+ New profile...";
        profileSelect.appendChild(newOption);

        const target = selectAfterLoad !== undefined ? selectAfterLoad : activeProfile;
        if (target && users.some(u => u.name === target)) {
            profileSelect.value = target;
        } else {
            profileSelect.value = "";
            activeProfile = "";
            localStorage.removeItem("pronunciation_profile");
        }
        updateProfileStatus();
    } catch (error) {
        console.error("Failed to load profiles:", error);
    }
}

function updateProfileStatus() {
    profileStatus.textContent = activeProfile ? `Practicing as "${activeProfile}"` : "";
}

profileSelect.onchange = async () => {
    const value = profileSelect.value;

    if (value === NEW_PROFILE_VALUE) {
        const name = prompt("Enter a name for your new profile:");
        if (!name || !name.trim()) {
            profileSelect.value = activeProfile;
            return;
        }
        try {
            const response = await fetch("/users", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: name.trim() }),
            });
            const data = await response.json();
            if (!response.ok || data.error) {
                alert(data.error || "Could not create profile.");
                profileSelect.value = activeProfile;
                return;
            }
            activeProfile = data.name;
            localStorage.setItem("pronunciation_profile", activeProfile);
            await loadProfiles(activeProfile);
        } catch (error) {
            alert("Request failed: " + error.message);
            profileSelect.value = activeProfile;
        }
        return;
    }

    activeProfile = value;
    if (activeProfile) {
        localStorage.setItem("pronunciation_profile", activeProfile);
    } else {
        localStorage.removeItem("pronunciation_profile");
    }
    updateProfileStatus();
};

loadProfiles();

// -----------------------------
// Tabs
// -----------------------------
const tabButtons = document.querySelectorAll(".tab-button");
const practiceWrapper = document.getElementById("practiceWrapper");
const adaptivePanel = document.getElementById("adaptivePanel");
const progressPanel = document.getElementById("progressPanel");
const textInput = document.getElementById("textInput");

tabButtons.forEach(button => {
    button.onclick = () => {
        tabButtons.forEach(b => b.classList.remove("active"));
        button.classList.add("active");
        const tab = button.dataset.tab;

        practiceWrapper.classList.toggle("hidden", tab === "progress");
        adaptivePanel.classList.toggle("hidden", tab !== "adaptive");
        progressPanel.classList.toggle("hidden", tab !== "progress");

        if (tab === "free") {
            currentSentenceId = null;
        }
        if (tab === "progress") {
            loadProgress();
        }
    };
});

textInput.addEventListener("input", () => {
    // Manual edits invalidate the link to whichever exercise was served --
    // otherwise a re-typed sentence would be silently credited to the
    // original adaptive exercise's assignment.
    currentSentenceId = null;
});

// -----------------------------
// Adaptive practice
// -----------------------------
const nextExerciseBtn = document.getElementById("nextExerciseBtn");
const adaptiveStatus = document.getElementById("adaptiveStatus");
const adaptiveExerciseInfo = document.getElementById("adaptiveExerciseInfo");
const adaptiveMode = document.getElementById("adaptiveMode");
const adaptiveTargets = document.getElementById("adaptiveTargets");

nextExerciseBtn.onclick = async () => {
    if (!activeProfile) {
        alert("Select or create a profile first.");
        return;
    }

    adaptiveStatus.textContent = "Fetching your next exercise...";
    adaptiveExerciseInfo.classList.add("hidden");

    try {
        const response = await fetch(`/practice/next?user=${encodeURIComponent(activeProfile)}`);
        const data = await response.json();

        if (!response.ok || data.error) {
            adaptiveStatus.textContent = data.error || "Could not fetch an exercise.";
            return;
        }

        textInput.value = data.text;
        currentSentenceId = data.sentence_id;

        adaptiveMode.textContent = describeMode(data.mode);
        adaptiveTargets.innerHTML = "";
        (data.target_phonemes || []).forEach(phoneme => {
            const badge = document.createElement("span");
            badge.className = "phoneme-badge";
            badge.textContent = phoneme;
            adaptiveTargets.appendChild(badge);
        });
        if (!data.target_phonemes || data.target_phonemes.length === 0) {
            adaptiveTargets.innerHTML = "<em>broad warm-up (no history yet)</em>";
        }

        adaptiveExerciseInfo.classList.remove("hidden");
        let statusLine = "Record yourself reading the sentence above, then click Analyze.";
        if (data.assessment) {
            const a = data.assessment;
            const score = a.pronunciation_score === null ? "n/a" : a.pronunciation_score;
            statusLine += ` Provisional level: ${a.overall_level} (${a.assessment_status}, score ${score}/100).`;
            if (data.exercise_type) {
                statusLine += ` Exercise type: ${data.exercise_type.replace(/_/g, " ")}.`;
            }
            if (data.confusion_hint) {
                statusLine += ` Focus: ${data.confusion_hint}.`;
            }
        }
        adaptiveStatus.textContent = statusLine;

        document.getElementById("results").classList.add("hidden");
        recordedBlob = null;
        analyzeBtn.disabled = true;
    } catch (error) {
        adaptiveStatus.textContent = "Request failed: " + error.message;
    }
};

function describeMode(mode) {
    if (mode === "diagnostic") return "Diagnostic (broad warm-up, not enough history yet)";
    if (mode === "generated") return "Freshly generated for your weak sounds";
    return "Targeted at your weakest sounds";
}

// -----------------------------
// Progress
// -----------------------------
const refreshProgressBtn = document.getElementById("refreshProgressBtn");
const progressStatus = document.getElementById("progressStatus");
const progressList = document.getElementById("progressList");
const historyList = document.getElementById("historyList");

refreshProgressBtn.onclick = loadProgress;

async function loadProgress() {
    if (!activeProfile) {
        progressStatus.textContent = "Select a profile above to see your per-phoneme progress.";
        progressList.innerHTML = "";
        historyList.innerHTML = "";
        return;
    }

    progressStatus.textContent = "Loading...";

    try {
        const [gapsResponse, historyResponse] = await Promise.all([
            fetch(`/practice/gaps?user=${encodeURIComponent(activeProfile)}`),
            fetch(`/practice/history?user=${encodeURIComponent(activeProfile)}`),
        ]);
        const gapsData = await gapsResponse.json();
        const historyData = await historyResponse.json();

        renderProgressList(gapsData.phonemes || []);
        renderHistoryList(historyData.attempts || []);

        progressStatus.textContent = (gapsData.phonemes || []).length
            ? `Showing ${gapsData.phonemes.length} tracked phoneme(s) for "${activeProfile}", weakest first.`
            : `No attempts recorded yet for "${activeProfile}". Try Adaptive Practice or Free Practice with a profile selected.`;
    } catch (error) {
        progressStatus.textContent = "Request failed: " + error.message;
    }
}

function renderProgressList(phonemes) {
    progressList.innerHTML = "";
    phonemes.forEach(item => {
        const row = document.createElement("div");
        row.className = "progress-row";

        const masteryPercent = Math.round(item.mastery * 100);
        const isWeak = item.lower_confidence_bound < 0.5;

        row.innerHTML = `
            <span class="progress-symbol">${escapeHtml(item.phoneme)}</span>
            <span class="progress-bar-track">
                <span class="progress-bar-fill ${isWeak ? "weak" : ""}" style="width: ${masteryPercent}%;"></span>
            </span>
            <span>${masteryPercent}%</span>
            <span class="progress-meta">${item.attempts_count} attempt(s)${item.example ? " &middot; e.g. " + escapeHtml(item.example) : ""}</span>
        `;
        progressList.appendChild(row);
    });
}

function renderHistoryList(attempts) {
    historyList.innerHTML = "";
    if (!attempts.length) {
        historyList.innerHTML = "<p class=\"empty-guide\">No attempts yet.</p>";
        return;
    }
    attempts.forEach(attempt => {
        const row = document.createElement("div");
        row.className = "history-row";
        row.innerHTML = `
            <span class="history-text">${escapeHtml(attempt.text)}</span>
            <span class="history-meta">PER ${attempt.phoneme_error_rate}% &middot; ${escapeHtml(attempt.created_at)}</span>
        `;
        historyList.appendChild(row);
    });
}
