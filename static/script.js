// Global variables for MediaRecorder and recording state
let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let currentStream = null; // To hold the MediaStream object for proper cleanup

// DOM element references (will be initialized on DOMContentLoaded)
let stopRecordingButton;
let recordingIndicator;
let leftPanel;
let rightPanel;
let toggleLeftBtn;
let toggleRightBtn;
let chatMessageInput;
let sendButton; // Reference to the 'Send' button
let micButton;  // Reference to the 'Microphone' button
let chatHistory;

// --- Constants ---
const CONSTANTS = {
    MAX_NAME_LENGTH: 80,
    CHAT_MODES: {
        GENERAL: 'general',
        DOCUMENT: 'document'
    },
    SPEECH_AUDIO_TYPE: 'audio/wav',
    MAX_DOC_NAME_LENGTH: 80 // Assuming same max length for docs
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

// --- NEW: Unified Message Display Function ---
function displayMessage(message, sender, messageId = null) {
    if (!chatHistory) return; // Ensure chatHistory element is available

    const messageElement = document.createElement('div');
    messageElement.classList.add('message-container');

    let displayRole = sender;
    let messageClass = '';

    if (sender === 'user') {
        displayRole = 'You';
        messageClass = 'user-message';
        messageElement.innerHTML = `
            <div class="message-content">
                <strong>${displayRole}:</strong> ${message}
            </div>
        `;
    } else if (sender === 'bot' || sender === 'assistant') {
        displayRole = 'Bot';
        messageClass = 'assistant-message';
        messageElement.innerHTML = `
            <div class="message-content">
                <strong>${displayRole}:</strong> <span class="bot-text-content">${message}</span>
                <button class="speaker-icon" data-text="${message}" aria-label="Play message">🔊</button>
            </div>
        `;
        // Attach event listener to the speaker icon immediately
        const speakerButton = messageElement.querySelector('.speaker-icon');
        if (speakerButton) {
            speakerButton.addEventListener('click', () => {
                const textToSpeak = speakerButton.dataset.text;
                playBotResponse(textToSpeak);
            });
        }
    } else { // For system messages, errors, etc.
        messageElement.innerHTML = `
            <div class="message-content">
                <strong>${displayRole}:</strong> ${message}
            </div>
        `;
    }

    messageElement.classList.add(messageClass);
    chatHistory.appendChild(messageElement);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Scroll to bottom
}


// Function to refresh the page to update chat/document lists
function refreshDashboard(chatId = null) {
    let url = "/dashboard";
    if (chatId) {
        url = `/dashboard?chat_id=${encodeURIComponent(chatId)}`;
    }
    window.location.href = url;
}

// Function to clean up microphone resources
function cleanupResources() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
        console.log("Microphone stream stopped and resources cleaned.");
    }
}

// --- NEW PANEL TOGGLING FUNCTIONS ---
function togglePanel(panelElement, storageKey, direction) {
    if (!panelElement) {
        console.warn('Panel element not found for toggling.');
        return;
    }
    const hiddenClass = `hidden-panel-${direction}`;
    panelElement.classList.toggle(hiddenClass);
    const isHidden = panelElement.classList.contains(hiddenClass);
    
    // Update button text/icon based on state
    const toggleButton = direction === 'left' ? toggleLeftBtn : toggleRightBtn;
    if (toggleButton) {
        toggleButton.textContent = isHidden ? (direction === 'left' ? '›' : '‹') : (direction === 'left' ? '‹' : '›');
    }

    // Store preference in localStorage
    localStorage.setItem(storageKey, isHidden ? 'true' : 'false');
    
    // Dispatch resize event if your main content needs to reflow
    window.dispatchEvent(new Event('resize'));
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
    if (trimmedName.length > CONSTANTS.MAX_NAME_LENGTH) {
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
    const msg = chatMessageInput.value.trim();
    if (!msg) return;

    const currentChatId = getAppData('chatId');
    if (!currentChatId) {
        alert("Please select or create a chat first.");
        return;
    }

    displayMessage(msg, "user"); // Use the unified displayMessage
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
        displayMessage(data.reply || "No response received.", "bot"); // Use the unified displayMessage
        
        const currentChatNameFromJinja = getAppData('chatName');
        if (currentChatNameFromJinja === "New Chat" && data.chat_id) {
            refreshDashboard(data.chat_id);
        }
    } catch (error) {
        console.error("Error sending message:", error);
        displayMessage("Sorry, an error occurred while sending your message. Please try again.", "bot"); // Use the unified displayMessage
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
            if (mode === CONSTANTS.CHAT_MODES.GENERAL) {
                generalBtn?.classList.add("bg-blue-600", "text-white");
                generalBtn?.classList.remove("bg-gray-300", "text-gray-800", "hover:bg-gray-400");
                documentBtn?.classList.remove("bg-blue-600", "text-white");
                documentBtn?.classList.add("bg-gray-300", "text-gray-800", "hover:bg-gray-400");
            } else {
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

async function playBotResponse(text) {
    try {
        const response = await fetch('/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        if (!response.ok) {
            const errorData = await response.json();
            console.error('TTS error:', errorData.error);
            alert('Could not play audio: ' + errorData.error);
            return;
        }

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();

        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
        };

    } catch (error) {
        console.error('Network or TTS playback error:', error);
        alert('An error occurred while playing the audio.');
    }
}


// --- Document Management Functions ---

// Handle document upload form submission
document.getElementById("upload-form")?.addEventListener("submit", async function(event) {
    event.preventDefault();

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
    if (submitButton) { // Null check for submitButton
        submitButton.disabled = true;
        submitButton.textContent = "Uploading...";
    }

    try {
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
        if (submitButton) { // Null check for submitButton
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
    if (trimmedName.length > CONSTANTS.MAX_DOC_NAME_LENGTH) {
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
            refreshDashboard(getAppData('chatId'));
        } else {
            alert("Failed to delete document: " + (data.error || "Unknown error"));
        }
    } catch (error) {
        console.error("Error deleting document:", error);
        alert("An error occurred while deleting the document.");
    }
}

async function toggleDocumentSelection(docId, selected, event) {
    const currentChatId = getAppData('chatId');
    if (!currentChatId) {
        alert("Please select a chat to manage document selection.");
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
            if (event && event.target) {
                event.target.checked = !selected;
            }
        } else {
            console.log(`Document ${docId} ${selected ? "select" : "deselect"}ed.`);
        }
    } catch (error) {
        console.error("Error toggling document selection:", error);
        alert("An error occurred while toggling document selection.");
        if (event && event.target) {
            event.target.checked = !selected;
        }
    }
}

// --- Speech Input Functions ---
async function startSpeechInput() {
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
    isRecording = true;

    // Disable send and microphone buttons
    sendButton?.setAttribute('disabled', 'true');
    micButton?.setAttribute('disabled', 'true');

    // Create and append recording indicator
    const chatInputAreaParent = safeQuerySelector('.p-4.bg-white.flex.space-x-3.border-t.shadow-inner');
    if (chatInputAreaParent) {
        recordingIndicator = document.createElement("span");
        recordingIndicator.className = "recording-indicator bg-red-500 text-white px-3 py-1 rounded-full text-sm ml-2";
        recordingIndicator.textContent = "Recording...";
        chatInputAreaParent.appendChild(recordingIndicator); 

        // Create and append stop button
        stopRecordingButton = document.createElement("button");
        stopRecordingButton.type = "button";
        stopRecordingButton.textContent = "Stop Recording";
        stopRecordingButton.className = "bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition duration-200 ml-2";
        stopRecordingButton.onclick = stopSpeechInput;
        chatInputAreaParent.appendChild(stopRecordingButton); 
    }


    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(currentStream);
        mediaRecorder.start();
        console.log("Recording started...");

        mediaRecorder.ondataavailable = event => {
            audioChunks.push(event.data);
        };

        mediaRecorder.onstop = async () => {
            isRecording = false;
            cleanupSpeechUI(); // Clean up UI elements

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
        // Re-enable send and mic buttons on error
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
    // Re-enable send and microphone buttons
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

document.addEventListener('DOMContentLoaded', () => {
    // Initialize DOM element references
    chatHistory = safeQuerySelector("#chat-history");
    chatMessageInput = safeQuerySelector("#chat-message");
    sendButton = safeQuerySelector("#send-message-btn"); // Assuming you add an ID to your send button
    micButton = safeQuerySelector("#microphone-btn");    // Assuming you add an ID to your microphone button

    leftPanel = safeQuerySelector('#left-panel');
    rightPanel = safeQuerySelector('#right-panel');
    toggleLeftBtn = safeQuerySelector('#toggle-left-panel-btn');
    toggleRightBtn = safeQuerySelector('#toggle-right-panel-btn');

    // Initial scroll to bottom for existing chat history on page load
    if (chatHistory) {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    const hasCurrentChat = getAppData('hasCurrentChat') === 'true';

    // Disable all relevant elements if no current chat
    document.querySelectorAll(
        '.top-controls button, ' +
        '#chat-message, ' + // Use ID for chat message input
        '#send-message-btn, ' + // Use ID for send button
        '#microphone-btn, ' + // Use ID for microphone button
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
    const currentChatId = getAppData('chatId');
    if (currentChatId) {
        const selectedChatItem = safeQuerySelector(`#chat-item-${currentChatId}`);
        if (selectedChatItem) {
            selectedChatItem.classList.add('selected');
        }
    }

    // Add click listeners to the toggle buttons
    toggleLeftBtn?.addEventListener('click', () => togglePanel(leftPanel, 'leftPanelHidden', 'left'));
    toggleRightBtn?.addEventListener('click', () => togglePanel(rightPanel, 'rightPanelHidden', 'right'));

    // Apply saved state from localStorage on page load
    if (localStorage.getItem('leftPanelHidden') === 'true') {
        leftPanel?.classList.add('hidden-panel-left');
        if (toggleLeftBtn) toggleLeftBtn.textContent = '›';
    }
    if (localStorage.getItem('rightPanelHidden') === 'true') {
        rightPanel?.classList.add('hidden-panel-right');
        if (toggleRightBtn) toggleRightBtn.textContent = '‹';
    }

    // --- IMPORTANT: Event listeners for buttons NOT handled by dynamic displayMessage or form submits ---
    safeQuerySelector('#new-chat-button')?.addEventListener('click', startNewChat);
    safeQuerySelector('#microphone-btn')?.addEventListener('click', startSpeechInput);
    safeQuerySelector('#send-message-btn')?.addEventListener('click', sendMessage);
    safeQuerySelector('#chat-message')?.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault(); // Prevent new line
            sendMessage();
        }
    });

    // Delegated event listener for chat management actions (rename, delete, resume)
    // This assumes chat actions are on elements with data-chat-id attributes inside #chat-list
    safeQuerySelector('#chat-list')?.addEventListener('click', (event) => {
        const target = event.target;
        const chatId = target.dataset.chatId || target.closest('[data-chat-id]')?.dataset.chatId;

        if (!chatId) return;

        if (target.classList.contains('rename-chat-btn')) {
            renameChat(chatId);
        } else if (target.classList.contains('delete-chat-btn')) {
            deleteChat(chatId);
        } else if (target.closest('.chat-list-item')) { // Assuming the whole item is clickable to resume
            resumeChat(chatId);
        }
    });

    // Delegated event listener for document management actions (rename, delete, toggle selection)
    // This assumes document actions are on elements with data-doc-id attributes inside #document-list
    safeQuerySelector('#document-list')?.addEventListener('click', (event) => {
        const target = event.target;
        const docId = target.dataset.docId || target.closest('[data-doc-id]')?.dataset.docId;

        if (!docId) return;

        if (target.classList.contains('rename-doc-btn')) {
            renameDocument(docId);
        } else if (target.classList.contains('delete-doc-btn')) {
            deleteDocument(docId);
        } else if (target.type === 'checkbox' && target.classList.contains('doc-select-checkbox')) {
            toggleDocumentSelection(docId, target.checked, event);
        }
    });

    // Event listeners for chat mode buttons
    safeQuerySelector('#general-mode-btn')?.addEventListener('click', () => {
        const currentChatId = getAppData('chatId');
        if (currentChatId) switchChatMode(currentChatId, CONSTANTS.CHAT_MODES.GENERAL);
        else alert("Please select or create a chat first.");
    });
    safeQuerySelector('#document-mode-btn')?.addEventListener('click', () => {
        const currentChatId = getAppData('chatId');
        if (currentChatId) switchChatMode(currentChatId, CONSTANTS.CHAT_MODES.DOCUMENT);
        else alert("Please select or create a chat first.");
    });
});
