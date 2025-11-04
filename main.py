"""LangGraph-driven PyBullet simulation.

This module launches a simple PyBullet simulation that contains a cube-based
robot and two colored balls. A LangGraph agent is used to plan and execute
high-level goals provided by the user via the command line.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

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
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, START, StateGraph


logger = logging.getLogger(__name__)

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
        self.blue_cube_id = self._create_blue_cube()
        self._yaw = 0.0
        self._heading_marker_id: int | None = None
        self._vision_llm: Optional[AzureChatOpenAI] = None

        self._time_step = 1.0 / 120.0
        p.setTimeStep(self._time_step)
        self._update_heading_marker()

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

    def _create_blue_cube(self) -> int:
        half_extents = [0.1, 0.1, 0.1]
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=half_extents
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.2, 0.2, 0.95, 1.0],
        )
        position = [0.6, 1.2, half_extents[2]]
        cube = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )
        return cube

    def get_robot_position(self) -> np.ndarray:
        position, _ = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(position)

    def get_ball_position(self, color: str) -> np.ndarray:
        if color not in self.balls:
            raise ValueError(f"Unknown ball color: {color}")
        position, _ = p.getBasePositionAndOrientation(self.balls[color])
        return np.array(position)

    def move_robot_towards(
        self,
        target: np.ndarray,
        tolerance: float = 0.05,
        step_distance: float = 0.1,
    ) -> bool:
        """Move the robot towards ``target`` in fixed increments."""

        if step_distance <= 0:
            raise ValueError("step_distance must be positive")

        max_steps = 600

        for _ in range(1, max_steps + 1):
            current = self.get_robot_position()
            delta = target - current
            planar_delta = delta[:2]
            distance = np.linalg.norm(planar_delta)

            if distance < tolerance:
                return True

            direction = planar_delta / max(distance, 1e-6)
            travel = min(step_distance, distance)
            step_xy = direction * travel
            new_position = np.array(
                [current[0] + step_xy[0], current[1] + step_xy[1], current[2]]
            )
            p.resetBasePositionAndOrientation(
                self.robot_id,
                new_position.tolist(),
                p.getQuaternionFromEuler([0, 0, self._yaw]),
            )
            self._update_heading_marker()
            p.stepSimulation()

            if self._use_gui:
                time.sleep(self._time_step)

            remaining = np.linalg.norm((target - new_position)[:2])

            if remaining < tolerance:
                return True

        return False

    def move_relative(self, delta: np.ndarray) -> str:
        current = self.get_robot_position()
        target = current + np.array([delta[0], delta[1], 0.0])
        arrived = self.move_robot_towards(target)
        if arrived:
            return "Movement command executed. Capture a new camera image to assess surroundings."
        return (
            "Movement halted before reaching the requested offset. Capture a camera image to reassess."
        )

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

    def move_forward(self, step_distance: float = 0.5) -> str:
        """Advance the robot forward by a fixed step."""

        if step_distance <= 0:
            raise ValueError("step_distance must be positive")
        summary = self.move_direction("forward", step_distance)
        return summary + f"\nForward step size: {step_distance:.2f}m."

    def turn(self, angle_degrees: float) -> str:
        self._yaw += math.radians(angle_degrees)
        # Normalize yaw to [-pi, pi]
        self._yaw = (self._yaw + math.pi) % (2 * math.pi) - math.pi
        position = self.get_robot_position()
        orientation = p.getQuaternionFromEuler([0.0, 0.0, self._yaw])
        p.resetBasePositionAndOrientation(self.robot_id, position.tolist(), orientation)
        self._update_heading_marker()
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

    def _angle_to_ball(self, ball_position: np.ndarray) -> float:
        robot_pos = self.get_robot_position()
        delta = ball_position[:2] - robot_pos[:2]
        return math.atan2(delta[1], delta[0])

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _update_heading_marker(self) -> None:
        """Display a visual indicator showing the robot's forward direction."""

        if self._heading_marker_id is not None:
            p.removeUserDebugItem(self._heading_marker_id)

        position = self.get_robot_position()
        heading_vector = np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
        marker_length = 0.4
        start = position
        end = position + heading_vector * marker_length
        self._heading_marker_id = p.addUserDebugLine(
            start.tolist(),
            end.tolist(),
            lineColorRGB=[0.9, 0.1, 0.1],
            lineWidth=3.0,
            lifeTime=0.0,
        )

    def set_detection_llm(self, llm: AzureChatOpenAI) -> None:
        """Configure the language model used for vision-based detection."""

        self._vision_llm = llm

    @staticmethod
    def _color_label(rgb: np.ndarray) -> str:
        r, g, b = rgb
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        if max_val < 50:
            return "K"  # Dark / background
        if min_val > 200:
            return "W"  # Bright / white
        if r > g + 50 and r > b + 50:
            return "R"
        if b > r + 50 and b > g + 50:
            return "B"
        if g > r + 50 and g > b + 50:
            return "G"
        if r + g > 420 and b < 160:
            return "Y"
        if b > 160 and g > 160 and r < 140:
            return "C"
        if r > 160 and b > 160 and g < 140:
            return "M"
        return "O"

    @staticmethod
    def _depth_buffer_to_meters(depth_buffer: np.ndarray, near: float, far: float) -> np.ndarray:
        return (2.0 * near * far) / (
            far + near - (2.0 * depth_buffer - 1.0) * (far - near)
        )

    def capture_and_detect(
        self, target_object: Optional[str] = None, fov_degrees: float = 60.0
    ) -> str:
        """Capture a virtual camera image and use an LLM to judge target visibility."""

        robot_pos = self.get_robot_position()
        camera_height = 0.2
        camera_offset = np.array([0.0, 0.0, camera_height])
        camera_origin = robot_pos + camera_offset
        heading = np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
        camera_target = camera_origin + heading
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_origin.tolist(),
            cameraTargetPosition=camera_target.tolist(),
            cameraUpVector=[0.0, 0.0, 1.0],
        )

        aspect_ratio = 1.0
        near_val = 0.02
        far_val = 5.0
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=fov_degrees,
            aspect=aspect_ratio,
            nearVal=near_val,
            farVal=far_val,
        )

        width = 128
        height = 128
        renderer = (
            p.ER_BULLET_HARDWARE_OPENGL if self._use_gui else p.ER_TINY_RENDERER
        )
        _, _, rgb_data, depth_data, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=renderer,
        )

        rgb_array = np.reshape(rgb_data, (height, width, 4)).astype(np.uint8)[..., :3]
        depth_buffer = np.array(depth_data, dtype=np.float32).reshape(height, width)
        depth_meters = self._depth_buffer_to_meters(depth_buffer, near_val, far_val)

        grid_size = 8
        block_h = height // grid_size
        block_w = width // grid_size
        trimmed_rgb = rgb_array[
            : grid_size * block_h, : grid_size * block_w
        ]
        block_means = trimmed_rgb.reshape(
            grid_size, block_h, grid_size, block_w, 3
        ).mean(axis=(1, 3))
        label_grid = [
            [self._color_label(block_means[row, col]) for col in range(grid_size)]
            for row in range(grid_size)
        ]
        grid_lines = ["".join(row) for row in label_grid]
        label_counts = Counter(label for row in label_grid for label in row)
        counts_summary = ", ".join(
            f"{label}:{count}" for label, count in sorted(label_counts.items())
        )

        summary_lines = [
            "Camera snapshot captured via RGB-D sensor.",
            (
                f"Robot pose -> position {robot_pos.round(3).tolist()}, "
                f"heading {math.degrees(self._yaw):.1f}°."
            ),
            (
                "Scene objects include red and blue balls plus a blue cube. "
                "All decisions must rely on the camera output."
            ),
            "Color grid (legend: R=red, B=blue, G=green, C=cyan, M=magenta, Y=yellow, "
            "W=bright, K=dark, O=other):",
        ]
        summary_lines.extend(f"   {line}" for line in grid_lines)
        summary_lines.append(
            f"Color counts: {counts_summary if counts_summary else 'none'}"
        )

        finite_depth = depth_meters[np.isfinite(depth_meters)]
        if finite_depth.size:
            summary_lines.append(
                (
                    "Depth statistics (meters): min "
                    f"{finite_depth.min():.2f}, mean {finite_depth.mean():.2f}, "
                    f"max {finite_depth.max():.2f}."
                )
            )
        else:
            summary_lines.append("Depth statistics unavailable (no valid depth samples).")

        target_object = (target_object or "").strip()
        if not target_object:
            summary_lines.append(
                "No target specified. Provide `target_object` to trigger vision analysis."
            )
            return "\n".join(summary_lines)

        if self._vision_llm is None:
            summary_lines.append(
                "Vision language model is not configured; cannot analyze target visibility."
            )
            return "\n".join(summary_lines)

        grid_text = "\n".join(grid_lines)
        counts_text = counts_summary if counts_summary else "none"
        detection_prompt = (
            "You analyze a coarse color grid derived from a robot-mounted camera. "
            "Determine whether the specified target object is visible. Respond with "
            "'yes', 'no', or 'uncertain' on the first line, followed by a brief "
            "justification."
            f"\nTarget object: {target_object}."
            "\nGrid legend: R=red, B=blue, G=green, C=cyan, M=magenta, Y=yellow, W=bright, K=dark, O=other."
            f"\nColor counts: {counts_text}."
            "\nGrid (top row = left-to-right across the forward view):\n"
            f"{grid_text}"
        )

        try:
            response = self._vision_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a concise vision classifier. Decide if the target object is visible "
                            "based on the provided color grid and counts."
                        )
                    ),
                    HumanMessage(content=detection_prompt),
                ]
            )
            detection_text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
        except Exception as exc:  # pragma: no cover - defensive
            detection_text = f"Vision model call failed: {exc}"

        summary_lines.append(
            f"Vision model assessment for '{target_object}': {detection_text.strip()}"
        )

        return "\n".join(summary_lines)

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
    def move_forward() -> str:
        """Move the robot 0.5m forward along its current heading."""

        return sim.move_forward(0.5)

    @tool
    def turn(angle_degrees: float) -> str:
        """Rotate the robot around the vertical axis by ``angle_degrees``."""

        return sim.turn(angle_degrees)

    @tool
    def capture_and_detect_image(target_object: str) -> str:
        """Capture a camera image and ask whether ``target_object`` is visible."""

        return sim.capture_and_detect(target_object=target_object)

    return [move_forward, turn, capture_and_detect_image]


def build_agent(sim: BulletSimulation, action_delay: float = 0.0):
    vision_llm = create_llm()
    sim.set_detection_llm(vision_llm)
    llm = create_llm()
    tools = build_tools(sim)
    tool_map = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    system_message = SystemMessage(
        content=(
            "You control a cube robot in a PyBullet world containing red and blue balls and a blue cube. "
            "Only three tools are available: ``turn`` (rotate by a degree value), ``move_forward`` "
            "(advance exactly 0.5m), and ``capture_and_detect_image`` (capture an RGB-D camera frame). "
            "Always call ``capture_and_detect_image`` with a ``target_object`` argument such as 'red ball', "
            "'blue ball', or 'blue cube'. The tool returns raw sensor summaries plus a vision-model "
            "judgement that must guide your decisions. Do not rely on hidden world knowledge—plan solely "
            "from camera observations. If the target is not yet visible, keep turning and capturing images. "
            "Once the target appears, alternate between capturing images and moving forward until you are "
            "very close (within roughly 0.2m). Provide concise progress updates throughout."
        )
    )

    graph = StateGraph(AgentState)

    def agent_node(state: AgentState) -> AgentState:
        logger.info("Agent node invoked with %d messages", len(state.messages))
        response = llm_with_tools.invoke(state.messages)
        requires_action = bool(getattr(response, "tool_calls", None))
        logger.info(
            "Agent response received. requires_action=%s tool_calls=%s",
            requires_action,
            getattr(response, "tool_calls", []),
        )
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
            logger.info("Executing tool '%s' with args=%s", tool_name, call.get("args"))
            if tool_name not in tool_map:
                content = f"Tool '{tool_name}' not found."
            else:
                tool_executor = tool_map[tool_name]
                try:
                    content = tool_executor.invoke(call["args"])
                    logger.info("Tool '%s' completed with result: %s", tool_name, content)
                    if action_delay > 0:
                        time.sleep(action_delay)
                except Exception as exc:  # pragma: no cover - defensive logging
                    content = f"Tool '{tool_name}' failed: {exc}"
                    logger.exception("Tool '%s' failed", tool_name)
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
        logger.info("Starting agent run for goal: %s", initial_state.goal)
        result = app.invoke(initial_state)
        if isinstance(result, dict):
            logger.info("Agent returned a raw dict; converting to AgentState")
            result = AgentState(**result)
        elif not isinstance(result, AgentState):
            raise TypeError(f"Unexpected agent result type: {type(result)!r}")
        logger.info(
            "Agent run complete. messages=%d requires_action=%s done=%s",
            len(result.messages),
            result.requires_action,
            result.done,
        )
        return result

    invoke.__doc__ = "Run the compiled graph while ensuring system instructions are present."

    class AgentWrapper:
        def __init__(self, runner):
            self._runner = runner

        def invoke(self, state: AgentState) -> AgentState:
            return self._runner(state)

    return AgentWrapper(invoke)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="LangGraph controlled PyBullet demo")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the simulation with the PyBullet GUI (if available).",
    )
    parser.add_argument(
        "--action-delay",
        type=float,
        default=0.0,
        help="Seconds to pause after each agent action (tool execution).",
    )
    args = parser.parse_args()

    sim = BulletSimulation(use_gui=args.gui)
    try:
        delay = max(args.action_delay, 0.0)
        if delay and not args.gui:
            logger.info("Action delay of %.2fs enabled for CLI mode.", delay)
        agent = build_agent(sim, action_delay=delay)
    except EnvironmentError as exc:
        sim.close()
        print("Azure OpenAI configuration error:", exc)
        logger.error("Azure OpenAI configuration error: %s", exc)
        return

    print("PyBullet simulation ready. Type a goal such as 'approach the red ball'.")
    print("Type 'quit' or press Ctrl+C to exit.\n")
    logger.info("Simulation initialized. Awaiting user goals.")

    try:
        while True:
            goal = input("Goal> ").strip()
            if not goal:
                continue
            if goal.lower() in {"quit", "exit"}:
                break

            logger.info("Processing goal: %s", goal)
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
            logger.info("Goal processing complete. Total messages: %d", len(final_state.messages))
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        logger.info("Shutdown requested by user.")
    finally:
        sim.close()
        logger.info("Simulation closed.")


if __name__ == "__main__":
    main()
