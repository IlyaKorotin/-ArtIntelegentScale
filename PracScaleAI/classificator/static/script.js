document.getElementById('send-button').addEventListener('click', sendMessage);

function sendMessage() {
    const messageInput = document.getElementById('message-input');
    const message = messageInput.value.trim();

    if (message) {
        const chatContent = document.getElementById('chat-content');
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.textContent = message;
        chatContent.appendChild(messageElement);

        // Очищаем поле ввода
        messageInput.value = '';
        // Прокручиваем чат до последнего сообщения
        chatContent.scrollTop = chatContent.scrollHeight;
    }
}
