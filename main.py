"""LangGraph-driven PyBullet simulation.

This module launches a simple PyBullet simulation that contains a cube-based
robot and two colored balls. A LangGraph agent is used to plan and execute
high-level goals provided by the user via the command line.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pybullet as p
import pybullet_data
from langchain.prompts import PromptTemplate
from langgraph.graph import END, START, StateGraph


PLAN_TEMPLATE = PromptTemplate.from_template(
    "Goal: {goal}\nTarget: {target}\nPlan:\n1. Identify the {target} ball\n2. Move to the {target} ball"
)


@dataclass
class AgentState:
    """Container for the agent state tracked across LangGraph nodes."""

    goal: str
    plan: List[str] = field(default_factory=list)
    current_step: int = 0
    target_color: Optional[str] = None
    target_position: Optional[np.ndarray] = None
    observation: str = ""
    done: bool = False
    status: str = ""


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
                self.robot_id, new_position.tolist(), [0, 0, 0, 1]
            )
            p.stepSimulation()

            if self._use_gui:
                time.sleep(self._time_step)

        return False

    def close(self) -> None:
        p.disconnect()


def parse_goal(goal: str) -> Optional[str]:
    goal_lower = goal.lower()
    if "red" in goal_lower:
        return "red"
    if "blue" in goal_lower:
        return "blue"
    return None


def build_agent(sim: BulletSimulation):
    graph = StateGraph(AgentState)

    def planner(state: AgentState) -> AgentState:
        target = state.target_color or parse_goal(state.goal)
        if not target:
            observation = (
                "Goal must mention either the red ball or the blue ball. "
                "Please provide a clearer instruction."
            )
            return AgentState(
                goal=state.goal,
                plan=[],
                current_step=0,
                target_color=None,
                observation=observation,
                status="Unable to determine target",
                done=True,
            )

        plan = [f"Identify the {target} ball", f"Move to the {target} ball"]
        plan_summary = PLAN_TEMPLATE.format(goal=state.goal, target=target)

        return AgentState(
            goal=state.goal,
            plan=plan,
            current_step=0,
            target_color=target,
            observation=plan_summary,
            status="Plan created",
            done=False,
        )

    def executor(state: AgentState) -> AgentState:
        if state.done:
            return state

        if state.current_step >= len(state.plan):
            return AgentState(
                goal=state.goal,
                plan=state.plan,
                current_step=state.current_step,
                target_color=state.target_color,
                target_position=state.target_position,
                observation=state.observation,
                status="Plan already completed",
                done=state.done,
            )

        step_description = state.plan[state.current_step]
        observation = state.observation
        done = state.done
        target_position = state.target_position
        status = state.status

        if "Identify" in step_description:
            if not state.target_color:
                observation = "Target color not set; cannot identify ball."
                done = True
                status = "Identification failed"
            else:
                target_position = sim.get_ball_position(state.target_color)
                observation = f"Located {state.target_color} ball at {target_position.round(3)}"
                status = "Target identified"
        elif "Move" in step_description:
            if target_position is None:
                observation = "No known target position. Need to identify the ball first."
                done = True
                status = "Movement aborted"
            else:
                success = sim.move_robot_towards(target_position)
                if success:
                    observation = (
                        f"Arrived near the {state.target_color} ball at {target_position.round(3)}"
                    )
                    status = "Reached destination"
                    done = True
                else:
                    observation = (
                        "Failed to reach the target within the allotted steps."
                    )
                    status = "Movement failed"
                    done = True
        else:
            observation = f"Unknown action: {step_description}"
            status = "Execution error"
            done = True

        return AgentState(
            goal=state.goal,
            plan=state.plan,
            current_step=state.current_step + 1,
            target_color=state.target_color,
            target_position=target_position,
            observation=observation,
            status=status,
            done=done,
        )

    def evaluator(state: AgentState) -> AgentState:
        if state.done:
            return state

        if state.current_step >= len(state.plan) and state.target_color:
            target_position = sim.get_ball_position(state.target_color)
            robot_position = sim.get_robot_position()
            distance = np.linalg.norm(target_position[:2] - robot_position[:2])
            if distance < 0.08:
                observation = (
                    f"Goal achieved. Robot is {distance:.3f} meters from the {state.target_color} ball."
                )
                return AgentState(
                    goal=state.goal,
                    plan=state.plan,
                    current_step=state.current_step,
                    target_color=state.target_color,
                    target_position=target_position,
                    observation=observation,
                    status="Goal achieved",
                    done=True,
                )

        return state

    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("evaluator", evaluator)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "evaluator")

    def done_condition(state: AgentState) -> bool:
        return state.done

    graph.add_conditional_edges(
        "evaluator",
        done_condition,
        {True: END, False: "executor"},
    )

    return graph.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph controlled PyBullet demo")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the simulation with the PyBullet GUI (if available).",
    )
    args = parser.parse_args()

    sim = BulletSimulation(use_gui=args.gui)
    agent = build_agent(sim)

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

            print(f"Status: {final_state.status}")
            print(f"Observation: {final_state.observation}\n")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
