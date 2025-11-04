# LangGraph PyBullet Simulation

This repository demonstrates how to combine [LangGraph](https://github.com/langchain-ai/langgraph) with
[LangChain](https://github.com/langchain-ai/langchain) to control a PyBullet simulation. A simple
LangGraph agent plans and executes goals such as moving a cube-shaped robot toward colored balls.

## Features

- PyBullet environment containing a plane, a cube robot, and two spheres (red and blue).
- LangGraph planning loop with planner, executor, and evaluator nodes.
- LangChain `PromptTemplate` used to summarize the generated plan.
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

```bash
python main.py
```

Pass `--gui` to launch the PyBullet GUI (if your environment supports it):

```bash
python main.py --gui
```

Once the program starts you can type instructions such as:

```
Goal> approach the red ball
```

The LangGraph agent will iteratively plan and execute until it reaches the requested target. Type
`quit` (or press `Ctrl+C`) to exit.
