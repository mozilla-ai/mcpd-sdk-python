"""Example demonstrating README generation using mcpd SDK with AI agents."""

from dotenv import load_dotenv
from mcpd import McpdClient
from pydantic_ai import Agent

load_dotenv()  # Load the API key for the agent (using Mistral in this case)

agent_model = "mistral:mistral-large-2411"

# Connect to the mcpd server to enable the agent access the tools running on it.
client = McpdClient(api_endpoint="http://localhost:8090")

agent = Agent(
    agent_model,
    tools=client.agent_tools(),
    system_prompt="You must use the available GitHub tool to search for a README file, "
    "create a new branch, create a new README file, and open a Pull Request.",
)

target_repo = input(
    "Enter the target GitHub repository (in the format owner/repo-name) where the new README should be created: "
)

input_repo_readme = input(
    "(Optional) Enter the GitHub repository (in the format owner/repo-name) to use its README "
    "as a template (press Enter to let the LLM define the structure): "
)

prompt = f"""
Create a new README file for the target repository ({target_repo}).

If a template README file from an original repository ({input_repo_readme}) is provided:

- Template Utilization: Use the structure and style of the original README file as a guide.
- Content Adaptation: Modify the content to accurately describe the target repository.
This step requires analyzing the entire codebase of the target repository ({target_repo})
to understand its functionality, features, and any other relevant details that should be included in the README.

If no template README file is provided:

README Creation: Generate a high-quality README file from scratch, including relevant sections such as:
-Project description and overview
-Installation and setup instructions
-Usage examples
-Configuration options
-Contributing guidelines
-License information
-Acknowledgements (if applicable)

To create the README, analyze the entire codebase of the target repository ({target_repo})
to understand its functionality, features, and any other relevant details.

After creating the new README file:

1. Branch Creation: Create a new branch in the target repository ({target_repo}) to hold the changes.
2. Commit: Commit the newly generated README file to this branch with a concise and descriptive
 commit message that clearly indicates the addition of the new README.
3. Pull Request (PR) Creation: Open a Pull Request against the main branch of the target
 repository, including the new README file.
4. PR Description: In the Pull Request description, mention that the README was generated
by an AI agent of the model {agent_model}. If a template was used, include a link to
the original README template ({input_repo_readme}) used for generating the new README.

When providing code snippets for this task, follow the guidelines for code modifications:

- Always specify the language and file name in the info string of code blocks (e.g., python src/main.py).
- For code modifications, use concise snippets that highlight the necessary changes, with abbreviated
  placeholders for unmodified sections.
- When editing existing files, restate the relevant function or class and use 'lazy' comments to omit
  unmodified portions.

Please provide a concise explanation of the changes made unless explicitly asked for code only
"""

agent_trace = agent.run_sync(prompt)
print(agent_trace.output)
