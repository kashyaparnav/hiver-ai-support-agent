(() => {
    const API_URL = "http://127.0.0.1:8000/api/chat";

    // =========================
    // Inject CSS
    // =========================

    const style = document.createElement("style");

    style.textContent = `
        #hiver-ai-widget {
            font-family: Arial, sans-serif;
        }

        #hiver-chat-button {
            position: fixed;
            right: 25px;
            bottom: 25px;
            width: 60px;
            height: 60px;
            border: none;
            border-radius: 50%;
            background: #111827;
            color: white;
            font-size: 25px;
            cursor: pointer;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
            z-index: 999999;
        }

        #hiver-chat-window {
            position: fixed;
            right: 25px;
            bottom: 95px;
            width: 360px;
            height: 520px;
            background: white;
            border-radius: 18px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
            display: none;
            flex-direction: column;
            overflow: hidden;
            z-index: 999999;
        }

        #hiver-chat-header {
            height: 65px;
            padding: 0 18px;
            background: #111827;
            color: white;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        #hiver-chat-header strong {
            display: block;
            font-size: 16px;
        }

        #hiver-chat-header small {
            display: block;
            margin-top: 3px;
            opacity: 0.75;
        }

        #hiver-chat-close {
            border: none;
            background: transparent;
            color: white;
            font-size: 25px;
            cursor: pointer;
        }

        #hiver-chat-messages {
            flex: 1;
            padding: 18px;
            overflow-y: auto;
            background: #f9fafb;
        }

        .hiver-message {
            max-width: 80%;
            padding: 11px 14px;
            margin-bottom: 12px;
            border-radius: 14px;
            font-size: 14px;
            line-height: 1.4;
        }

        .hiver-bot-message {
            background: white;
            color: #111827;
            border: 1px solid #e5e7eb;
            margin-right: auto;
        }

        .hiver-user-message {
            background: #111827;
            color: white;
            margin-left: auto;
        }

        #hiver-chat-input-area {
            display: flex;
            padding: 12px;
            border-top: 1px solid #e5e7eb;
            background: white;
        }

        #hiver-chat-input {
            flex: 1;
            border: 1px solid #d1d5db;
            border-radius: 10px;
            padding: 11px;
            outline: none;
            font-size: 14px;
        }

        #hiver-chat-input:focus {
            border-color: #111827;
        }

        #hiver-chat-send {
            width: 45px;
            margin-left: 8px;
            border: none;
            border-radius: 10px;
            background: #111827;
            color: white;
            cursor: pointer;
            font-size: 18px;
        }

        #hiver-chat-send:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        @media (max-width: 500px) {
            #hiver-chat-window {
                right: 10px;
                bottom: 80px;
                width: calc(100% - 20px);
                height: 70vh;
            }

            #hiver-chat-button {
                right: 15px;
                bottom: 15px;
            }
        }
    `;

    document.head.appendChild(style);


    // =========================
    // Create Widget HTML
    // =========================

    const widget = document.createElement("div");

    widget.id = "hiver-ai-widget";

    widget.innerHTML = `
        <button id="hiver-chat-button">
            💬
        </button>

        <div id="hiver-chat-window">

            <div id="hiver-chat-header">
                <div>
                    <strong>🤖 AI Support</strong>
                    <small>Online</small>
                </div>

                <button id="hiver-chat-close">
                    ×
                </button>
            </div>

            <div id="hiver-chat-messages">

                <div class="hiver-message hiver-bot-message">
                    Hi! 👋 How can I help you today?
                </div>

            </div>

            <div id="hiver-chat-input-area">

                <input
                    id="hiver-chat-input"
                    type="text"
                    placeholder="Type your message..."
                />

                <button id="hiver-chat-send">
                    ➤
                </button>

            </div>

        </div>
    `;

    document.body.appendChild(widget);


    // =========================
    // Get Elements
    // =========================

    const chatButton =
        document.getElementById("hiver-chat-button");

    const chatWindow =
        document.getElementById("hiver-chat-window");

    const closeButton =
        document.getElementById("hiver-chat-close");

    const sendButton =
        document.getElementById("hiver-chat-send");

    const input =
        document.getElementById("hiver-chat-input");

    const messages =
        document.getElementById("hiver-chat-messages");


    // =========================
    // Open Chat
    // =========================

    chatButton.addEventListener("click", () => {
        chatWindow.style.display = "flex";
        chatButton.style.display = "none";
        input.focus();
    });


    // =========================
    // Close Chat
    // =========================

    closeButton.addEventListener("click", () => {
        chatWindow.style.display = "none";
        chatButton.style.display = "flex";
    });


    // =========================
    // Add Message
    // =========================

    function addMessage(text, type) {

        const message =
            document.createElement("div");

        message.className =
            type === "user"
                ? "hiver-message hiver-user-message"
                : "hiver-message hiver-bot-message";

        message.textContent = text;

        messages.appendChild(message);

        messages.scrollTop =
            messages.scrollHeight;
    }


    // =========================
    // Send Message
    // =========================

    async function sendMessage() {

        const message =
            input.value.trim();

        if (!message) {
            return;
        }

        addMessage(message, "user");

        input.value = "";

        sendButton.disabled = true;

        addMessage("Typing...", "bot");


        try {

            const response =
                await fetch(API_URL, {

                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        message: message,
                        top_k: 3
                    })
                });


            const data =
                await response.json();


            // Remove typing message

            const typingMessage =
                messages.lastElementChild;

            if (typingMessage) {
                typingMessage.remove();
            }


            // Handle response

            if (data.success) {

                addMessage(
                    data.response,
                    "bot"
                );


                // Ticket created after escalation

                if (data.ticket) {

                    addMessage(
                        `🎫 Ticket created: ${data.ticket.ticket_id}. Our human support team will take it from here.`,
                        "bot"
                    );

                }

            } else {

                addMessage(
                    "Sorry, something went wrong. Please try again.",
                    "bot"
                );

            }

        } catch (error) {

            const typingMessage =
                messages.lastElementChild;

            if (typingMessage) {
                typingMessage.remove();
            }

            addMessage(
                "Unable to connect to the support agent.",
                "bot"
            );

            console.error(
                "Hiver AI Widget Error:",
                error
            );

        } finally {

            sendButton.disabled = false;

            input.focus();

        }
    }


    // =========================
    // Send Button
    // =========================

    sendButton.addEventListener(
        "click",
        sendMessage
    );


    // =========================
    // Enter Key
    // =========================

    input.addEventListener(
        "keydown",
        (event) => {

            if (event.key === "Enter") {
                sendMessage();
            }

        }
    );

})();