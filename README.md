# LangGraph PyBullet Simulation

This repository demonstrates how to combine [LangGraph](https://github.com/langchain-ai/langgraph) with
[LangChain](https://github.com/langchain-ai/langchain) to control a PyBullet simulation. A simple
LangGraph agent plans and executes goals such as moving a cube-shaped robot toward colored balls.

## Features

- PyBullet environment containing a plane, a cube robot, and two spheres (red and blue).
- LangGraph planning loop powered by an Azure OpenAI chat model.
- Tool-enabled agent that can move, turn, and evaluate the simulated scene through
  registered LangChain tools.
- Interactive CLI: provide goals like "approach the red ball" and observe the agent's progress.

## Setup

Create a virtual environment and install the latest LangGraph, LangChain, and PyBullet releases:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install langgraph langchain pybullet numpy
```

## Running the simulation

Before running the simulation, export the Azure OpenAI environment variables so the
agent can reach your deployment:

```bash
export OPENAI_API_KEY="<your-azure-openai-key>"
export OPENAI_API_VERSION="<api-version>"
export AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com"
export OPENAI_MODEL_NAME="<deployment-name>"
```

Then start the simulator:

```bash
python main.py        # add --gui if you want the PyBullet GUI
```

Pass `--action-delay <seconds>` to pause briefly after each agent tool execution.

Pass `--gui` to launch the PyBullet GUI (if your environment supports it).

Once the program starts you can type instructions such as:

```
Goal> approach the red ball
```

The LangGraph agent will iteratively plan and execute until it reaches the requested target. Type
`quit` (or press `Ctrl+C`) to exit.
