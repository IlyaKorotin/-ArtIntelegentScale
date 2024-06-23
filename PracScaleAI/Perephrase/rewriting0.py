from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

api_key = "JLoFuUPZm4dpsUKP5JCmO8Ei5i8WELtK"
model = "mistral-large-latest"

def Rewriting(input_text):
    client = MistralClient(api_key=api_key)

    messages = [
        ChatMessage(role="user", content=f"Перефразируй текст: {input_text}")
    ]

    chat_response = client.chat(
        model=model,
        messages=messages,
    )

    return chat_response.choices[0].message.content

input_text=input()
print(Rewriting(input_text))