// Global variables for MediaRecorder and recording state
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let stopRecordingButton;
let recordingIndicator;
let currentStream = null; // New: To hold the MediaStream object for proper cleanup

// --- Constants (New: for magic numbers/strings) ---
const CONSTANTS = {
    MAX_NAME_LENGTH: 80,
    CHAT_MODES: {
        GENERAL: 'general',
        DOCUMENT: 'document'
    },
    SPEECH_AUDIO_TYPE: 'audio/wav'
};

// --- Helper Functions ---

// New: Safe DOM query selector
function safeQuerySelector(selector) {
    const element = document.querySelector(selector);
    if (!element) {
        console.warn(`Element not found: ${selector}`);
    }
    return element;
}

// Helper to get CSRF token from meta tag
function getCsrfToken() {
    const metaTag = safeQuerySelector('meta[name="csrf-token"]');
    return metaTag ? metaTag.content : '';
}

// New: Read app data safely from HTML data attributes
function getAppData(key) {
    const appDataEl = safeQuerySelector('#app-data');
    return appDataEl ? appDataEl.dataset[key] : '';
}

// Helper for safely appending messages to chat history
function appendMsg(role, text) {
    const chatHistory = safeQuerySelector("#chat-history");
    if (!chatHistory) return;

    const wrapper = document.createElement("div");
    const strong = document.createElement("strong");

    let displayRole = role;
    let messageClass = '';
    if (role === 'user') {
        displayRole = 'You';
        messageClass = 'user-message';
    } else if (role === 'bot' || role === 'assistant') {
        displayRole = 'Bot';
        messageClass = 'assistant-message';
    }

    strong.textContent = displayRole + ":";
    wrapper.appendChild(strong);
    // FIX: Use textContent for full safety, avoid direct string append that could be XSS
    const textNode = document.createTextNode(" " + text);
    wrapper.appendChild(textNode);
    wrapper.classList.add(messageClass);

    chatHistory.appendChild(wrapper);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Function to refresh the page to update chat/document lists
function refreshDashboard(chatId = null) {
    let url = "/dashboard";
    if (chatId) {
        url = `/dashboard?chat_id=${encodeURIComponent(chatId)}`;
    }
    window.location.href = url;
}

// New: Function to clean up microphone resources
function cleanupResources() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
        console.log("Microphone stream stopped and resources cleaned.");
    }
}

// --- Chat Management Functions ---
async function renameChat(chatId) {
    const newName = prompt("Enter new chat name (max 80 chars):");
    if (newName === null) return;
    const trimmedName = newName.trim();
    if (trimmedName === '') {
        alert("Chat name cannot be empty.");
        return;
    }
    if (trimmedName.length > CONSTANTS.MAX_NAME_LENGTH) { // Use constant
        alert(`Chat name cannot exceed ${CONSTANTS.MAX_NAME_LENGTH} characters.`);
        return;
    }

    try {
        const params = new URLSearchParams({ chat_id: chatId, new_name: trimmedName });
        const response = await fetch("/rename_chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRF-Token": getCsrfToken()
            },
            body: params.toString()
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === "success") {
            refreshDashboard(chatId);
        } else {
            alert("Failed to rename chat: " + (data.error || "Unknown error"));
        }
    } catch (error) {
        console.error("Error renaming chat:", error);
        alert("An error occurred while renaming the chat.");
    }
}

async function deleteChat(chatId) {
    if (!confirm("Are you sure you want to delete this chat? This action cannot be undone.")) {
        return;
    }

    try {
        const params = new URLSearchParams({ chat_id: chatId });
        const response = await fetch("/delete_chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRF-Token": getCsrfToken()
            },
            body: params.toString()
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === "deleted") {
            refreshDashboard();
        } else {
            alert("Failed to delete chat: " + (data.error || "Unknown error"));
        }
    } catch (error) {
        console.error("Error deleting chat:", error);
        alert("An error occurred while deleting the chat.");
    }
}

function resumeChat(chatId) {
    refreshDashboard(chatId);
}

async function startNewChat() {
    try {
        const response = await fetch("/start_chat", {
            method: "POST",
            headers: {
                "X-CSRF-Token": getCsrfToken()
            }
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === "success" && data.chat_id) {
            refreshDashboard(data.chat_id);
        } else {
            alert("Failed to start new chat: " + (data.error || "Unknown error"));
        }
    } catch (error) {
        console.error("Error starting new chat:", error);
        alert("An error occurred while starting a new chat.");
    }
}

// --- Chat Interaction Functions ---
async function sendMessage() {
    const chatMessageInput = safeQuerySelector("#chat-message");
    const msg = chatMessageInput.value.trim();
    if (!msg) return;

    // FIX: Read currentChatId safely from data attribute
    const currentChatId = getAppData('chatId');
    if (!currentChatId) {
        alert("Please select or create a chat first.");
        return;
    }

    appendMsg("user", msg);
    chatMessageInput.value = "";
    chatMessageInput.style.height = 'auto';

    try {
        const params = new URLSearchParams({ chat_id: currentChatId, message: msg });
        const response = await fetch("/send_message", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRF-Token": getCsrfToken()
            },
            body: params.toString()
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        appendMsg("assistant", data.reply || "No response received.");
        
        // FIX: Read currentChatNameFromJinja safely from data attribute
        const currentChatNameFromJinja = getAppData('chatName');
        if (currentChatNameFromJinja === "New Chat" && data.chat_id) {
             refreshDashboard(data.chat_id);
        }
    } catch (error) {
        console.error("Error sending message:", error);
        appendMsg("assistant", "Sorry, an error occurred while sending your message. Please try again.");
    }
}

async function switchChatMode(chatId, mode) {
    try {
        const params = new URLSearchParams({ chat_id: chatId, mode: mode });
        const response = await fetch("/switch_chat_mode", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRF-Token": getCsrfToken()
            },
            body: params.toString()
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === "mode updated") {
            const generalBtn = safeQuerySelector("#general-mode-btn");
            const documentBtn = safeQuerySelector("#document-mode-btn");
            if (mode === CONSTANTS.CHAT_MODES.GENERAL) { // Use constant
                generalBtn?.classList.add("bg-blue-600", "text-white");
                generalBtn?.classList.remove("bg-gray-300", "text-gray-800", "hover:bg-gray-400");
                documentBtn?.classList.remove("bg-blue-600", "text-white");
                documentBtn?.classList.add("bg-gray-300", "text-gray-800", "hover:bg-gray-400");
            } else { // Use constant
                documentBtn?.classList.add("bg-blue-600", "text-white");
                documentBtn?.classList.remove("bg-gray-300", "text-gray-800", "hover:bg-gray-400");
                generalBtn?.classList.remove("bg-blue-600", "text-white");
                generalBtn?.classList.add("bg-gray-300", "text-gray-800", "hover:bg-gray-400");
            }
            console.log(`Switched to ${mode} mode.`);
        } else {
            alert("Failed to switch chat mode: " + (data.error || "Unknown error"));
        }
    } catch (error) {
        console.error("Error switching chat mode:", error);
        alert("An error occurred while switching chat mode.");
    }
}

// --- Document Management Functions ---

// Handle document upload form submission
document.getElementById("upload-form")?.addEventListener("submit", async function(event) {
    event.preventDefault();

    // FIX: Read currentChatId safely from data attribute
    const currentChatId = getAppData('chatId');
    if (!currentChatId) {
        alert("Please select or create a chat before uploading documents.");
        return;
    }

    const fileInput = this.querySelector('input[type="file"]');
    if (fileInput.files.length === 0) {
        alert("Please select a file to upload.");
        return;
    }

    const submitButton = this.querySelector('button[type="submit"]');
    // FIX: Add null checks for submitButton
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.textContent = "Uploading...";
    }

    try { // FIX: Added try-catch block for robustness
        const formData = new FormData(this);
        formData.append("chat_id", currentChatId);
        formData.append("csrf_token", getCsrfToken());

        const response = await fetch("/upload_document", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === "uploaded") {
            alert(`Document "${data.filename}" uploaded and processed successfully!`);
            fileInput.value = '';
            refreshDashboard(currentChatId);
        } else {
            alert(`Failed to upload document: ${data.error || 'Unknown error'}`);
        }
    } catch (error) {
        console.error("Error uploading document:", error);
        alert("An error occurred during document upload.");
    } finally {
        // FIX: Add null checks for submitButton
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.textContent = "Upload Document";
        }
    }
});

async function renameDocument(docId) {
    const newName = prompt("Enter new document name (max 80 chars):");
    if (newName === null) return;
    const trimmedName = newName.trim();
    if (trimmedName === '') {
        alert("Document name cannot be empty.");
        return;
    }
    if (trimmedName.length > CONSTANTS.MAX_DOC_NAME_LENGTH) { // Use constant (assuming same max length)
        alert(`Document name cannot exceed ${CONSTANTS.MAX_DOC_NAME_LENGTH} characters.`);
        return;
    }

    try {
        const params = new URLSearchParams({ doc_id: docId, new_name: trimmedName });
        const response = await fetch("/rename_document", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRF-Token": getCsrfToken()
            },
            body: params.toString()
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === "renamed") {
            // FIX: Read currentChatId safely from data attribute
            refreshDashboard(getAppData('chatId'));
        } else {
            alert("Failed to rename document: " + (data.error || "Unknown error"));
        }
    } catch (error) {
        console.error("Error renaming document:", error);
        alert("An error occurred while renaming the document.");
    }
}

async function deleteDocument(docId) {
    if (!confirm("Are you sure you want to delete this document and all its associated data? This cannot be undone.")) {
        return;
    }

    try {
        const params = new URLSearchParams({ doc_id: docId });
        const response = await fetch("/delete_document", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRF-Token": getCsrfToken()
            },
            body: params.toString()
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === "deleted") {
            // FIX: Read currentChatId safely from data attribute
            refreshDashboard(getAppData('chatId'));
        } else {
            alert("Failed to delete document: " + (data.error || "Unknown error"));
        }
    } catch (error) {
        console.error("Error deleting document:", error);
        alert("An error occurred while deleting the document.");
    }
}

// FIX: Added 'event' parameter for toggleDocumentSelection
async function toggleDocumentSelection(docId, selected, event) {
    // FIX: Read currentChatId safely from data attribute
    const currentChatId = getAppData('chatId');
    if (!currentChatId) {
        alert("Please select a chat to manage document selection.");
        // FIX: Ensure event and target exist before reverting
        if (event && event.target) {
            event.target.checked = !selected;
        }
        return;
    }
    try {
        const params = new URLSearchParams({ chat_id: currentChatId, doc_id: docId, action: (selected ? "select" : "deselect") });
        const response = await fetch("/toggle_document", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
                "X-CSRF-Token": getCsrfToken()
            },
            body: params.toString()
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status !== "updated") {
            alert("Failed to update document selection: " + (data.error || "Unknown error"));
            // FIX: Ensure event and target exist before reverting
            if (event && event.target) {
                event.target.checked = !selected;
            }
        } else {
            console.log(`Document ${docId} ${selected ? "select" : "deselect"}ed.`);
        }
    } catch (error) {
        console.error("Error toggling document selection:", error);
        alert("An error occurred while toggling document selection.");
        // FIX: Ensure event and target exist before reverting
        if (event && event.target) {
            event.target.checked = !selected;
        }
    }
}

// --- Speech Input Functions ---
async function startSpeechInput() {
    // FIX: Read currentChatId safely from data attribute
    const currentChatId = getAppData('chatId');
    if (!currentChatId) {
        alert("Please select or create a chat first.");
        return;
    }
    if (isRecording) {
        alert("Already recording. Click 'Stop Recording' to finish.");
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Speech input is not supported by your browser. Please use a modern browser like Chrome or Firefox.");
        return;
    }

    audioChunks = [];
    isRecording = true; // Set recording state immediately

    const chatInputArea = safeQuerySelector('.chat-input');
    const sendButton = safeQuerySelector('.chat-input button:not(.microphone-btn)');
    const micButton = safeQuerySelector('.microphone-btn');

    // Disable send and microphone buttons (with null checks)
    sendButton?.setAttribute('disabled', 'true');
    micButton?.setAttribute('disabled', 'true');

    // Create and append recording indicator
    recordingIndicator = document.createElement("span");
    recordingIndicator.className = "recording-indicator bg-red-500 text-white px-3 py-1 rounded-full text-sm ml-2";
    recordingIndicator.textContent = "Recording...";
    chatInputArea?.appendChild(recordingIndicator); // Use optional chaining

    // Create and append stop button
    stopRecordingButton = document.createElement("button");
    stopRecordingButton.type = "button";
    stopRecordingButton.textContent = "Stop Recording";
    stopRecordingButton.className = "bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition duration-200 ml-2";
    stopRecordingButton.onclick = stopSpeechInput;
    chatInputArea?.appendChild(stopRecordingButton); // Use optional chaining

    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ audio: true }); // Store stream globally
        mediaRecorder = new MediaRecorder(currentStream);
        mediaRecorder.start();
        console.log("Recording started...");

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            isRecording = false;
            cleanupSpeechUI(); // Clean up UI elements

            // Use constant for audio type
            const audioBlob = new Blob(audioChunks, { type: CONSTANTS.SPEECH_AUDIO_TYPE });
            const formData = new FormData();
            formData.append('audio', audioBlob, 'audio.wav');
            formData.append("csrf_token", getCsrfToken());

            console.log("Sending audio for transcription...");
            try {
                const response = await fetch('/speech_input', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                const chatMessageInput = safeQuerySelector("#chat-message"); // Re-query or cache
                if (data.transcribed_text && chatMessageInput) {
                    chatMessageInput.value = data.transcribed_text;
                    alert("Transcription complete. Review the text and click Send.");
                } else {
                    alert("Transcription failed: " + (data.error || "Unknown error"));
                }
            } catch (error) {
                console.error("Error during speech input fetch:", error);
                alert("An error occurred during speech input processing.");
            } finally {
                cleanupResources(); // Guaranteed cleanup of microphone stream
            }
        };
    } catch (error) {
        isRecording = false; // Ensure state is reset on error
        cleanupSpeechUI(); // Clean up UI elements
        cleanupResources(); // Ensure microphone access is stopped
        console.error("Error accessing microphone:", error);
        alert("Error accessing microphone. Please ensure permissions are granted and no other application is using it.");
        // Re-enable send and mic buttons on error (with null checks)
        sendButton?.removeAttribute('disabled');
        micButton?.removeAttribute('disabled');
    }
}

function stopSpeechInput() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        console.log("Recording stopped by user.");
    }
}

function cleanupSpeechUI() {
    if (recordingIndicator && recordingIndicator.parentNode) {
        recordingIndicator.parentNode.removeChild(recordingIndicator);
        recordingIndicator = null;
    }
    if (stopRecordingButton && stopRecordingButton.parentNode) {
        stopRecordingButton.parentNode.removeChild(stopRecordingButton);
        stopRecordingButton = null;
    }
    // Re-enable send and microphone buttons (with null checks)
    const sendButton = safeQuerySelector('.chat-input button:not(.microphone-btn)');
    const micButton = safeQuerySelector('.microphone-btn');
    sendButton?.removeAttribute('disabled');
    micButton?.removeAttribute('disabled');
}

// --- Event Listeners and Initial Setup ---

// Auto-adjust textarea height
document.addEventListener('input', function (event) {
    if (event.target && event.target.tagName.toLowerCase() === 'textarea' && event.target.id === 'chat-message') {
        event.target.style.height = 'auto';
        event.target.style.height = (event.target.scrollHeight) + 'px';
    }
});

// Initial scroll to bottom for existing chat history on page load
document.addEventListener('DOMContentLoaded', () => {
    const chatHistory = safeQuerySelector("#chat-history");
    if (chatHistory) {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    // FIX: Read hasCurrentChat safely from data attribute
    const hasCurrentChat = getAppData('hasCurrentChat') === 'true';

    // Disable all relevant elements if no current chat
    document.querySelectorAll(
        '.top-controls button, ' +
        '.chat-input textarea, ' +
        '.chat-input button, ' +
        '#upload-form input, ' +
        '#upload-form button, ' +
        '#document-list input[type="checkbox"], ' +
        '#document-list .rename-btn, ' +
        '#document-list .delete-btn, ' +
        '.chat-mode-buttons button'
    ).forEach(el => {
        if (!hasCurrentChat && el.id !== 'new-chat-button') {
            el.disabled = true;
        }
    });

    // Ensure chat items are visually selected
    // FIX: Read currentChatId safely from data attribute
    const currentChatId = getAppData('chatId');
    if (currentChatId) {
        const selectedChatItem = safeQuerySelector(`#chat-item-${currentChatId}`);
        if (selectedChatItem) {
            selectedChatItem.classList.add('selected');
        }
    }
});