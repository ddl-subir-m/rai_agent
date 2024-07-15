# AI Code Analyzer

This project is an AI-powered code analysis system that uses Retrieval-Augmented Generation (RAG) to provide insights into AI systems based on their source code.

## Project Structure

```
ai-code-analyzer/
│
├── main.py
├── rag_system.py
├── agents.py
├── config.py
├── utils.py
├── requirements.txt
├── .env
└── README.md
```

## File Descriptions

### main.py
The entry point of the application. It sets up the RAG system, initializes the agents, and manages the conversation flow.

Key features:
- Configurable code path and recursive search options
- Initializes the RAG system with the specified code files
- Sets up the conversation agents (CodeAnalysisAgent, UserProxyAgent, FunctionCallingAgent)
- Manages the conversation and saves the results

### rag_system.py
Implements the Retrieval-Augmented Generation (RAG) system for code analysis.

Key components:
- Code embedding generation using CodeBERT
- Chunking and indexing of code files
- RAG search functionality

### agents.py
Defines the custom agents used in the conversation system.

Agents:
- CodeAnalysisAgent: Analyzes code and provides insights
- FunctionCallingAgent: Manages function calls within the conversation
- UserProxyAgent: Represents the user in the conversation

### config.py
Contains configuration variables for the project.

Key configurations:
- EMPTY_MESSAGE_LIMIT: Defines when to terminate the conversation
- LLM_CONFIG: Configuration for the language model (e.g., GPT-4)

### utils.py
Utility functions used across the project.

Key function:
- chunk_code: Splits code into manageable chunks for analysis

### requirements.txt
Lists all the Python dependencies required to run the project.

### .env
Stores environment variables, such as API keys. (Note: This file should not be committed to version control)

## Setup and Usage

1. Clone the repository:
   ```
   git clone https://github.com/ddl-subir-m/rai_agent.git
   cd rai_agent
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the project root
   - Add your OpenAI API key: `OPENAI_API_KEY=your_api_key_here`

4. Run the analysis:
   ```
   python main.py [--code_path PATH] [--recursive BOOL]
   ```
   - `--code_path`: Glob pattern for code files to analyze (default: "code/**/*.py")
   - `--recursive`: Whether to search for files recursively (default: True)

5. The analysis results will be saved in `conversation_result.json`.

## Customization

You can customize the analysis by modifying the following:
- Adjust the RAG system parameters in `rag_system.py`
- Modify the agent behaviors in `agents.py`
- Update the LLM configuration in `config.py`

