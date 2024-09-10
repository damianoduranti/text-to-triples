import os
from dotenv import load_dotenv
from openai import AzureOpenAI

class AzureOpenAIClient:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self._validate_env_vars()
        self._configure_openai_api()

    def _validate_env_vars(self):
        """Validate that all required environment variables are set."""
        required_vars = ['AZURE_OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'OPENAI_API_VERSION', 'AZURE_OPENAI_DEPLOYMENT_NAME']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise Exception(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _configure_openai_api(self):
        """Configure the Azure OpenAI client using environment variables."""
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
            api_version=os.getenv("OPENAI_API_VERSION")
        )

    def send_request(self, context, question):
        """
        Send a request to the Azure OpenAI API with the provided context and question.
        
        Parameters:
            context (str): The context to be included in the system message.
            question (str): The user's question to be sent to the API.
        
        Returns:
            response: The API response if successful, otherwise None.
        """
        try:
            # Sending a chat completion request
            response = self.client.chat.completions.create(
                model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": question}
                ]
            )
            return response
        except Exception as e:
            print(f"Failed request to API: {e}")
            return None


# Example usage
if __name__ == '__main__':
    context = "This is a test."
    question = "Respond with 'test'."
    
    prompting = AzureOpenAIClient()
    response = prompting.send_request(context, question)
    
    if response:
        # Printing the response choices
        print(response.choices[0].message.content)
        print("\n\n\n")
        print(response)
    else:
        print("No response received from the API.")