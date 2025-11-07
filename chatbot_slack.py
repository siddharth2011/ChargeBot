import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_ollama import OllamaLLM
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

llm = OllamaLLM(model="llama2", temperature=0.7)

# Use in-memory chat message history for each user
user_histories = {}

def get_user_history(user_id):
    if user_id not in user_histories:
        user_histories[user_id] = InMemoryChatMessageHistory()
    return user_histories[user_id]

conversation = RunnableWithMessageHistory(
    runnable=llm,
    get_session_history=lambda session_id: get_user_history(session_id),
)

app = App(token=os.environ["SLACK_BOT_TOKEN"])

@app.message("")  # Listen to all messages
def handle_message(message, say):
    user_input = message.get("text", "")
    user_id = message.get("user", "default")
    if user_input:
        try:
            response = conversation.invoke({"input": user_input}, config={"configurable": {"session_id": user_id}})
            say(response)
        except Exception as e:
            say(f"Error: {e}")

if __name__ == "__main__":
    handler = SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    handler.start()
