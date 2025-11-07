from langchain_community.llms import Ollama
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.conversation.base import ConversationChain

import os

def main():
    # Initialize the LLM with Ollama (opensource)
    llm = Ollama(
        model="llama2",
        temperature=0.7
    )
    
    # Initialize conversation memory
    memory = ConversationBufferMemory()
    
    # Create conversation chain
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    
    print("Chatbot initialized! Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            response = conversation.predict(input=user_input)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
