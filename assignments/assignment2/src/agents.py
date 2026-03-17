import os
from datetime import datetime
from openai import OpenAI

# Using OpenAI API
# Using gpt-5-nano
'''
Class to represent a debate agent
Class accepts role_name, api_key, model_name, system_prompt_path, temperature, and max_tokens as arguments
Class initializes the openAI client within the class so each instance has its own connection
system_prompt_path is stored as an instance variable
temperature is stored as an instance variable
max_tokens is stored as an instance variable
'''
class DebateAgent:
    #__init__ method
    #initialize openAI() client using the key from .env file
    #Accept a role_name and a system_instruction
    #Hyperparameters: Store the model, temperate and max_tokens as instance variables
    #Initialize an empty list called messages in OpenAI's format    
    def __init__(self, role_name, model_name, api_key=None,
                 system_prompt_path="", temperature=0.7, max_tokens=100):
        self.role_name = role_name
        self.model_name = model_name
        
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        
        self.system_prompt_path = system_prompt_path
        self.system_prompt = ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.messages = []
        self.current_response = ""
        self.timestamp = None

    def load_prompt_template(self, filepath):
        """Reads the raw string from a prompt template file."""
        try:
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading template: {e}")
            return ""

    '''
    Internal Message Formatter
    Role Management: Create a helper that converts the system_instruction into a message with the role "system"
    History Integration: Ensure that every time a new round happens, the opponent's argument is added to this
    messages list with the role "user"
    Context Preservation: Ensure that the history is preserved across rounds chronologically
    '''
    def _format_messages(self):
        # Simply return the persistent messages list (system prompt is added in __init__)
        return self.messages

    def add_to_history(self, role, content):
        """Helper to append to conversation tracking memory."""
        self.messages.append({"role": role, "content": content})
        
    '''
    generate_response method
    API Call: Use client.chat.completions.create()
    Structure: The messages parameter should be your formatted list (System Message + all previous turns)
    Pass temperature and max tokens from instance variables
    Chain-of-Thought: This method should be prepared to handle the full text block returned by the model
    Error Handling: Add a check for empty responses or API connection issues to ensure the program doesn't crash
    '''
    def generate_response(self, question, opponent_argument=None, custom_transcript=None):
        try:
            # Add opponent's argument to history if provided
            if opponent_argument:
                self.add_to_history("user", opponent_argument)
            elif not any(msg["role"] == "user" for msg in self.messages):
                # OpenAI requires at least one 'user' message to trigger a response
                self.add_to_history("user", "Please analyze the provided context and output your response following all constraints.")
                
            # Transcript Generation: Use custom transcript if provided (e.g., for Judge), otherwise build internally
            if custom_transcript is not None:
                transcript = custom_transcript
            else:
                # Skip the first message if it's the system prompt
                transcript_messages = self.messages[1:] if self.messages and self.messages[0].get("role") == "system" else self.messages
                
                # Format messages into a string
                transcript = ""
                for msg in transcript_messages:
                    # Add a label based on the role to make it readable in the prompt
                    speaker = "Assistant" if msg["role"] == "assistant" else "Opponent"
                    transcript += f"\\n{speaker}: {msg['content']}"
            
            # 1. Load the template
            raw_template = self.load_prompt_template(self.system_prompt_path)
            
            # 2. Variable Injection
            formatted_prompt = raw_template.replace("{{QUESTION}}", question).replace("{{TRANSCRIPT}}", transcript)
            self.system_prompt = formatted_prompt
            
            # 3. System Prompt Update in the persistent messages memory
            if not self.messages:
                self.messages.append({"role": "system", "content": self.system_prompt})
            elif self.messages[0]["role"] == "system":
                self.messages[0]["content"] = self.system_prompt
            else:
                self.messages.insert(0, {"role": "system", "content": self.system_prompt})
            
            # Call OpenAI API with persistent memory
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages
            )
            
            # Extract response and timestamp
            self.current_response = response.choices[0].message.content.strip()
            self.timestamp = datetime.now().isoformat()
            
            # Add agent's response to its own history
            self.add_to_history("assistant", self.current_response)
            
            return self.current_response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return None
    
    '''
    to dictionary export method
    create a method that captures the state of a single turn
    Should return a dictionary containing role_name, content of LLM's repsonse, the model name, and a timestamp
    '''
    def to_dict(self):
        return {
            "role": self.role_name,
            "content": self.current_response,
            "model": self.model_name,
            "timestamp": self.timestamp
        }
