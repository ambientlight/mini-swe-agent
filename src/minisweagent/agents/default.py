"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation."""

import json
import re
import subprocess
import time
from typing import Literal

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model


class AgentConfig(BaseModel):
    # Check the config files in minisweagent/config for example settings
    system_template: str
    instance_template: str
    timeout_template: str
    format_error_template: str
    action_observation_template: str
    action_regex: str = r"```bash\s*\n(.*?)\n```"
    action_parser: Literal["regex", "json"] = "regex"
    """Parser to use for extracting actions. 'regex' uses action_regex, 'json' parses JSON with 'command' field."""
    fallback_action_regex: str | None = None
    """Fallback regex tried when action_regex finds 0 matches. Allows graceful recovery from format drift (e.g. model outputs markdown instead of XML tags)."""
    json_command_field: str = "command"
    """Field name containing the command when using json action_parser."""
    step_limit: int = 0
    cost_limit: float = 3.0
    max_consecutive_errors: int | None = 10
    """Maximum consecutive format errors (including empty commands) before terminating. Set to null to disable the gate."""
    tolerant_multi_action: bool = False
    """When True, take the first action if multiple are found instead of raising FormatError."""
    loop_breaker: bool = False
    """When True, inject a warning after 3 identical consecutive commands to break loops."""


class NonTerminatingException(Exception):
    """Raised for conditions that can be handled by the agent."""


class FormatError(NonTerminatingException):
    """Raised when the LM's output is not in the expected format."""


class ExecutionTimeoutError(NonTerminatingException):
    """Raised when the action execution timed out."""


class TerminatingException(Exception):
    """Raised for conditions that terminate the agent."""


class Submitted(TerminatingException):
    """Raised when the LM declares that the agent has finished its task."""


class LimitsExceeded(TerminatingException):
    """Raised when the agent has reached its cost or step limit."""


class EmptyActionError(NonTerminatingException):
    """Raised when the LM provides an empty action."""


class TooManyConsecutiveErrors(TerminatingException):
    """Raised when the agent has too many consecutive format/empty action errors."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self._consecutive_errors = 0

    def render_template(self, template: str, **kwargs) -> str:
        template_vars = self.config.model_dump() | self.env.get_template_vars() | self.model.get_template_vars()
        return Template(template, undefined=StrictUndefined).render(
            **kwargs, **template_vars, **self.extra_template_vars
        )

    def add_message(self, role: str, content: str, **kwargs):
        self.messages.append({"role": role, "content": content, "timestamp": time.time(), **kwargs})

    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self._consecutive_errors = 0
        self._recent_actions: list[str] = []
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        while True:
            try:
                self.step()
                self._consecutive_errors = 0  # Reset on successful step
            except (FormatError, EmptyActionError) as e:
                self._consecutive_errors += 1
                if (
                    self.config.max_consecutive_errors is not None
                    and self._consecutive_errors >= self.config.max_consecutive_errors
                ):
                    raise TooManyConsecutiveErrors(
                        f"Agent terminated after {self._consecutive_errors} consecutive format/empty action errors. "
                        f"Last error: {e}"
                    )
                self.add_message("user", str(e))
            except NonTerminatingException as e:
                self._consecutive_errors = 0  # Reset on other non-terminating exceptions
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        return self.get_observation(self.query())

    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        self.add_message("assistant", **response)
        return response

    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)

        # Loop breaker: detect 3+ identical consecutive commands
        if self.config.loop_breaker:
            action_str = output.get("action", "").strip()
            recent = getattr(self, "_recent_actions", [])
            recent.append(action_str)
            if len(recent) > 5:
                recent.pop(0)
            self._recent_actions = recent

            if len(recent) >= 3 and len(set(recent[-3:])) == 1:
                observation += (
                    "\n\nLOOP DETECTED: Your last 3 commands were identical. "
                    "You MUST try a fundamentally different approach NOW. Options:\n"
                    "- Look at different files or different parts of the code\n"
                    "- Try a completely different fix strategy\n"
                    "- If you believe your fix is correct, submit immediately with:\n"
                    "  cd /testbed && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"
                )
                self._recent_actions = []

        self.add_message("user", observation)
        return output

    def parse_action(self, response: dict) -> dict:
        """Parse the action from the message. Returns the action."""
        if self.config.action_parser == "json":
            return self._parse_action_json(response)
        return self._parse_action_regex(response)

    def _parse_action_regex(self, response: dict) -> dict:
        """Parse action using regex, with optional fallback regex."""
        actions = re.findall(self.config.action_regex, response["content"], re.DOTALL)
        # If primary regex found nothing, try fallback before raising error
        if len(actions) == 0 and self.config.fallback_action_regex:
            actions = re.findall(self.config.fallback_action_regex, response["content"], re.DOTALL)
        if len(actions) == 1 or (len(actions) > 1 and self.config.tolerant_multi_action):
            # Take first action; tolerant_multi_action skips FormatError on multiple tags
            action = actions[0].strip()
            if not action:
                raise EmptyActionError(
                    "Empty command provided. Please provide a valid bash command to execute, "
                    "or if you have completed the task, use the submission command as specified in the instructions."
                )
            return {"action": action, **response}
        raise FormatError(self.render_template(self.config.format_error_template, actions=actions))

    def _parse_action_json(self, response: dict) -> dict:
        """Parse action from JSON response."""
        content = response["content"].strip()

        # Handle markdown-wrapped JSON (```json ... ``` or ``` ... ```)
        json_block_match = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", content, re.DOTALL)
        if json_block_match:
            content = json_block_match.group(1).strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise FormatError(
                self.render_template(self.config.format_error_template, actions=[])
                + f"\n\nJSON parse error: {e}"
            )

        command_field = self.config.json_command_field
        if command_field not in data:
            raise FormatError(
                self.render_template(self.config.format_error_template, actions=[])
                + f"\n\nMissing required field: '{command_field}'"
            )

        action = data[command_field]
        if not isinstance(action, str):
            raise FormatError(
                self.render_template(self.config.format_error_template, actions=[])
                + f"\n\nField '{command_field}' must be a string, got {type(action).__name__}"
            )

        action = action.strip()
        if not action:
            raise EmptyActionError(
                "Empty command provided. Please provide a valid bash command to execute, "
                "or if you have completed the task, use the submission command as specified in the instructions."
            )

        return {"action": action, **response}

    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action["action"])
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output | {"action": action["action"]}

    def has_finished(self, output: dict[str, str]):
        """Raises Submitted exception with final output if the agent has finished its task."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines:
            first_line = lines[0].strip()
            # Exact matches
            if first_line in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                raise Submitted("".join(lines[1:]))
            # Common typos (e.g., "SUMIT" instead of "SUBMIT")
            if first_line.startswith("COMPLETE_TASK_AND_SU") and first_line.endswith("_FINAL_OUTPUT"):
                raise Submitted("".join(lines[1:]))
