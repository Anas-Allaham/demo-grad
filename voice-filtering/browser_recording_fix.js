/*
Drop-in recording helper for the web app.
Goal: preserve the raw voice as much as possible before Python filtering.

Important changes:
1. Disable browser-level echo cancellation, noise suppression, and auto gain control.
2. Collect all MediaRecorder chunks, including the final chunk after stop().
3. Upload the raw Blob to the backend before any filtering/noise reduction.
*/

let mediaStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;

async function startRecording() {
  recordedChunks = [];
  recordedBlob = null;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
    video: false,
  });

  const track = mediaStream.getAudioTracks()[0];
  console.log("Actual microphone settings:", track.getSettings());

  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : "";

  mediaRecorder = mimeType
    ? new MediaRecorder(mediaStream, { mimeType })
    : new MediaRecorder(mediaStream);

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      recordedChunks.push(event.data);
    }
  };

  mediaRecorder.onerror = (event) => {
    console.error("MediaRecorder error:", event.error || event);
  };

  mediaRecorder.onstop = () => {
    const type = mediaRecorder.mimeType || "audio/webm";
    recordedBlob = new Blob(recordedChunks, { type });

    const audioPlayer = document.getElementById("audioPlayer");
    if (audioPlayer) {
      audioPlayer.src = URL.createObjectURL(recordedBlob);
      audioPlayer.controls = true;
    }

    console.log("Recorded blob:", {
      type: recordedBlob.type,
      size: recordedBlob.size,
      chunks: recordedChunks.length,
    });

    mediaStream.getTracks().forEach((t) => t.stop());
  };

  // Start without a tiny timeslice. This avoids creating many small chunks.
  mediaRecorder.start();
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
}

async function uploadRecording(url = "/analyze") {
  if (!recordedBlob) {
    throw new Error("No recording found. Record first, then upload.");
  }

  const formData = new FormData();
  formData.append("audio", recordedBlob, "raw_browser_recording.webm");

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
  }

  return await response.json();
}
