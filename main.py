"""LangGraph-driven PyBullet simulation.

This module launches a simple PyBullet simulation that contains a cube-based
robot and two colored balls. A LangGraph agent is used to plan and execute
high-level goals provided by the user via the command line.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np
import pybullet as p
import pybullet_data
import dotenv
dotenv.load_dotenv()
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langgraph.graph import END, START, StateGraph


PLAN_TEMPLATE = (
    "Goal: {goal}\n"
    "Target: {target}\n"
    "Plan:\n"
    "1. Identify the {target} ball\n"
    "2. Move to the {target} ball"

)
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph

@dataclass
class AgentState:
    """Container for the conversational agent state."""

    goal: str
    messages: List[BaseMessage] = field(default_factory=list)
    requires_action: bool = False
    done: bool = False


class BulletSimulation:
    """Utility wrapper around PyBullet for this demonstration."""

    def __init__(self, use_gui: bool = False):
        self._use_gui = use_gui
        connection_mode = p.GUI if use_gui else p.DIRECT
        self._client = p.connect(connection_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self._plane_id = p.loadURDF("plane.urdf")
        self.robot_id = self._create_robot()
        self.balls = self._create_balls()
        self._yaw = 0.0

        self._time_step = 1.0 / 120.0
        p.setTimeStep(self._time_step)

    def _create_robot(self) -> int:
        half_extents = [0.1, 0.1, 0.1]
        mass = 1.0
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=half_extents, rgbaColor=[0.1, 0.8, 0.1, 1.0]
        )
        position = [0.0, -1.0, 0.1]
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        robot = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            baseOrientation=orientation,
        )
        return robot

    def _create_balls(self) -> Dict[str, int]:
        radius = 0.1
        collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        balls = {}

        for color_name, rgba, position in [
            ("red", [0.9, 0.2, 0.2, 1.0], [1.0, 0.5, radius]),
            ("blue", [0.2, 0.2, 0.9, 1.0], [-1.0, 0.5, radius]),
        ]:
            visual_shape = p.createVisualShape(
                p.GEOM_SPHERE, radius=radius, rgbaColor=rgba
            )
            body_id = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position,
            )
            balls[color_name] = body_id

        return balls

    def get_robot_position(self) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(position)

    def get_ball_position(self, color: str) -> np.ndarray:
        if color not in self.balls:
            raise ValueError(f"Unknown ball color: {color}")
        position, _ = p.getBasePositionAndOrientation(self.balls[color])
        return np.array(position)

    def move_robot_towards(self, target: np.ndarray, tolerance: float = 0.05) -> bool:
        """Move the robot towards ``target`` using a kinematic controller."""

        max_steps = 600
        speed = 1.2  # meters per second

        for _ in range(max_steps):
            current = self.get_robot_position()
            delta = target - current
            distance = np.linalg.norm(delta[:2])

            if distance < tolerance:
                return True

            direction = delta[:2] / max(distance, 1e-6)
            step_xy = direction * speed * self._time_step
            new_position = np.array([current[0] + step_xy[0], current[1] + step_xy[1], current[2]])
            p.resetBasePositionAndOrientation(
                self.robot_id,
                new_position.tolist(),
                p.getQuaternionFromEuler([0, 0, self._yaw]),
            )
            p.stepSimulation()

            if self._use_gui:
                time.sleep(self._time_step)

        return False

    def move_relative(self, delta: np.ndarray) -> str:
        current = self.get_robot_position()
        target = current + np.array([delta[0], delta[1], 0.0])
        arrived = self.move_robot_towards(target)
        if arrived:
            return f"Moved to {target.round(3).tolist()}"
        return "Movement incomplete within allotted steps"

    def move_direction(self, direction: str, distance: float) -> str:
        if distance <= 0:
            return "Distance must be positive."

        direction = direction.lower()
        base_heading = self._yaw
        if direction == "forward":
            heading = base_heading
        elif direction == "backward":
            heading = base_heading + math.pi
        elif direction == "left":
            heading = base_heading + math.pi / 2
        elif direction == "right":
            heading = base_heading - math.pi / 2
        else:
            return (
                "Unknown direction. Use one of: forward, backward, left, right."
            )

        delta = np.array([math.cos(heading), math.sin(heading), 0.0]) * distance
        return self.move_relative(delta)

    def turn(self, angle_degrees: float) -> str:
        self._yaw += math.radians(angle_degrees)
        # Normalize yaw to [-pi, pi]
        self._yaw = (self._yaw + math.pi) % (2 * math.pi) - math.pi
        position = self.get_robot_position()
        orientation = p.getQuaternionFromEuler([0.0, 0.0, self._yaw])
        p.resetBasePositionAndOrientation(self.robot_id, position.tolist(), orientation)
        p.stepSimulation()
        return f"Turned to heading {math.degrees(self._yaw):.1f} degrees."

    def evaluate_scene(self) -> str:
        robot_pos = self.get_robot_position()
        details = [
            f"Robot at {robot_pos.round(3).tolist()}"
        ]
        for color, body_id in self.balls.items():
            ball_pos = p.getBasePositionAndOrientation(body_id)[0]
            ball_pos_arr = np.array(ball_pos)
            distance = np.linalg.norm(ball_pos_arr[:2] - robot_pos[:2])
            details.append(
                f"{color.title()} ball at {ball_pos_arr.round(3).tolist()} (distance {distance:.2f}m)"
            )
        return "; ".join(details)

    def close(self) -> None:
        p.disconnect()


def create_llm() -> AzureChatOpenAI:
    required_env = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_API_VERSION": os.getenv("OPENAI_API_VERSION"),
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME"),
    }

    missing = [name for name, value in required_env.items() if not value]
    if missing:
        missing_str = ", ".join(missing)
        raise EnvironmentError(
            "Missing required Azure OpenAI environment variables: " f"{missing_str}"
        )

    return AzureChatOpenAI(
        azure_endpoint=required_env["AZURE_OPENAI_ENDPOINT"],
        azure_deployment=required_env["OPENAI_MODEL_NAME"],
        openai_api_key=required_env["OPENAI_API_KEY"],
        openai_api_version=required_env["OPENAI_API_VERSION"],
        temperature=0.0,
    )


def build_tools(sim: BulletSimulation):
    @tool
    def move(direction: Literal["forward", "backward", "left", "right"], distance: float) -> str:
        """Move the robot in the given direction by ``distance`` meters."""

        return sim.move_direction(direction, distance)

    @tool
    def turn(angle_degrees: float) -> str:
        """Rotate the robot around the vertical axis by ``angle_degrees``."""

        return sim.turn(angle_degrees)

    @tool
    def evaluate_scene_image() -> str:
        """Provide a textual evaluation of the current simulated camera image."""

        return sim.evaluate_scene()

    return [move, turn, evaluate_scene_image]


def build_agent(sim: BulletSimulation):
    llm = create_llm()
    tools = build_tools(sim)
    tool_map = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    system_message = SystemMessage(
        content=(
            "You control a cube robot in a PyBullet world containing red and blue balls. "
            "Use the available tools to move, rotate, and analyze the scene so that you can "
            "reach goals provided by the user. Respond with clear, concise updates when the "
            "goal is satisfied."
        )
    )

    graph = StateGraph(AgentState)

    def agent_node(state: AgentState) -> AgentState:
        response = llm_with_tools.invoke(state.messages)
        requires_action = bool(getattr(response, "tool_calls", None))
        messages = state.messages + [response]
        return AgentState(
            goal=state.goal,
            messages=messages,
            requires_action=requires_action,
            done=not requires_action,
        )

    def tool_node(state: AgentState) -> AgentState:
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            return state

        tool_messages: List[BaseMessage] = []
        for call in last_message.tool_calls:
            tool_name = call["name"]
            if tool_name not in tool_map:
                content = f"Tool '{tool_name}' not found."
            else:
                tool_executor = tool_map[tool_name]
                try:
                    content = tool_executor.invoke(call["args"])
                except Exception as exc:  # pragma: no cover - defensive logging
                    content = f"Tool '{tool_name}' failed: {exc}"
            tool_messages.append(
                ToolMessage(content=content, tool_call_id=call["id"], name=tool_name)
            )

        messages = state.messages + tool_messages
        return AgentState(
            goal=state.goal,
            messages=messages,
            requires_action=False,
            done=False,
        )

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_edge("tools", "agent")

    def should_continue(state: AgentState) -> bool:
        return state.requires_action

    graph.add_conditional_edges("agent", should_continue, {True: "tools", False: END})

    app = graph.compile()

    def invoke(state: AgentState) -> AgentState:
        if not state.messages:
            initial_state = AgentState(
                goal=state.goal,
                messages=[system_message, HumanMessage(content=state.goal)],
                requires_action=False,
            )
        else:
            initial_state = state
        return app.invoke(initial_state)

    invoke.__doc__ = "Run the compiled graph while ensuring system instructions are present."

    class AgentWrapper:
        def __init__(self, runner):
            self._runner = runner

        def invoke(self, state: AgentState) -> AgentState:
            return self._runner(state)

    return AgentWrapper(invoke)


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph controlled PyBullet demo")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the simulation with the PyBullet GUI (if available).",
    )
    args = parser.parse_args()

    sim = BulletSimulation(use_gui=args.gui)
    try:
        agent = build_agent(sim)
    except EnvironmentError as exc:
        sim.close()
        print("Azure OpenAI configuration error:", exc)
        return

    print("PyBullet simulation ready. Type a goal such as 'approach the red ball'.")
    print("Type 'quit' or press Ctrl+C to exit.\n")

    try:
        while True:
            goal = input("Goal> ").strip()
            if not goal:
                continue
            if goal.lower() in {"quit", "exit"}:
                break

            state = AgentState(goal=goal)
            final_state = agent.invoke(state)

            print()
            for message in final_state.messages:
                if isinstance(message, (SystemMessage, HumanMessage)):
                    continue
                if isinstance(message, ToolMessage):
                    print(f"[Tool:{message.name}] {message.content}")
                elif isinstance(message, AIMessage):
                    if isinstance(message.content, str) and message.content.strip():
                        print(f"Assistant: {message.content.strip()}")
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
