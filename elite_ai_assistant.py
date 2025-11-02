"""
Elite Multi-AI Screen Assistant - Top Tier Models Only
Supports GPT-5, Claude 4.5 Sonnet, Claude 4 Opus (via OpenRouter), and Grok 4
"""

import asyncio
import json
import base64
import io
import threading
import time
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Screen capture and control
import pyautogui
import mss
from PIL import Image

# Process monitoring
import psutil
import subprocess
import queue
from collections import deque

# Web server and WebSocket
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# AI Integration
import requests
from openai import OpenAI
import anthropic

# Configure pyautogui for smoother control
pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    XAI = "xai"
    GROQ = "groq"
    TOGETHER = "together"

@dataclass
class AIModel:
    """Configuration for an AI model"""
    provider: AIProvider
    model_id: str
    display_name: str
    description: str
    requires_api_key: str
    supports_vision: bool
    icon: str
    color: str

# Elite AI Models Only - The Best of the Best
AI_MODELS = {
    "gpt-5": AIModel(
        provider=AIProvider.OPENAI,
        model_id="gpt-5",
        display_name="GPT-5",
        description="OpenAI's most advanced model",
        requires_api_key="openai",
        supports_vision=True,
        icon="🧠",
        color="#00a67e"
    ),
    "claude-4-5-sonnet": AIModel(
        provider=AIProvider.ANTHROPIC,
        model_id="claude-sonnet-4-5-20250929",
        display_name="Claude 4.5 Sonnet",
        description="Anthropic's latest and greatest",
        requires_api_key="anthropic",
        supports_vision=True,
        icon="⚡",
        color="#d97706"
    ),
    "claude-4-opus": AIModel(
        provider=AIProvider.OPENROUTER,
        model_id="anthropic/claude-4-opus",
        display_name="Claude 4 Opus",
        description="Most powerful Claude via OpenRouter",
        requires_api_key="openrouter",
        supports_vision=True,
        icon="👑",
        color="#7c3aed"
    ),
    "grok-4": AIModel(
        provider=AIProvider.XAI,
        model_id="xai/grok-4",
        display_name="Grok 4",
        description="xAI's latest - Unfiltered intelligence",
        requires_api_key="xai",
        supports_vision=True,
        icon="🚀",
        color="#dc2626"
    ),
    # Budget Models - For continuous oversight
    "gpt-4o-mini": AIModel(
        provider=AIProvider.OPENAI,
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        description="Budget OpenAI - Fast & cheap for monitoring",
        requires_api_key="openai",
        supports_vision=True,
        icon="💰",
        color="#10b981"
    ),
    "claude-haiku": AIModel(
        provider=AIProvider.ANTHROPIC,
        model_id="claude-3-5-haiku-20241022",
        display_name="Claude Haiku",
        description="Budget Claude - Lightning fast",
        requires_api_key="anthropic",
        supports_vision=True,
        icon="⚡",
        color="#22c55e"
    ),
    "groq-llama": AIModel(
        provider=AIProvider.GROQ,
        model_id="llama-3.1-70b-versatile",
        display_name="Llama 3.1 (Groq)",
        description="Ultra-fast inference - Perfect for oversight",
        requires_api_key="groq",
        supports_vision=False,
        icon="🐎",
        color="#8b5cf6"
    ),
    "together-llama": AIModel(
        provider=AIProvider.TOGETHER,
        model_id="meta-llama/Llama-3-70b-chat-hf",
        display_name="Llama 3 (Together)",
        description="Fast & cheap - Continuous monitoring",
        requires_api_key="together",
        supports_vision=False,
        icon="🤝",
        color="#06b6d4"
    )
}

# Error patterns to detect in terminal output
ERROR_PATTERNS = {
    'npm_lifecycle': r'ERR!.*ELIFECYCLE',
    'npm_resolve': r'ERR!.*ERESOLVE',
    'npm_enoent': r'ERR!.*ENOENT',
    'npm_econnrefused': r'ERR!.*ECONNREFUSED',
    'pip_error': r'ERROR:.*pip',
    'python_error': r'(Traceback|Exception|Error):',
    'docker_error': r'ERROR:.*docker|docker:.*Error',
    'psql_error': r'ERROR:.*psql|FATAL:.*database',
    'mongo_error': r'Error:.*MongoDB|MongoError',
    'sqlite_busy': r'SQLITE_BUSY|database is locked',
    'migration_failed': r'migration.*failed|error.*migrat',
    'build_failed': r'build.*failed|compilation.*error',
    'test_failed': r'test.*failed|FAIL:|✗',
    'port_in_use': r'EADDRINUSE|port.*already in use',
}

# Command whitelist - only these commands are allowed to be executed
SAFE_COMMANDS = {
    # Package managers
    'npm': [r'^npm (install|i|ci|update|uninstall|run .+)( .+)?$'],
    'pnpm': [r'^pnpm (install|add|update|remove|run .+)( .+)?$'],
    'yarn': [r'^yarn (install|add|upgrade|remove|run .+)( .+)?$'],
    'pip': [r'^pip install .+$', r'^pip uninstall .+$', r'^pip list$', r'^pip show .+$'],
    'python': [r'^python -m pip .+$'],
    'py': [r'^py -m pip .+$'],

    # Docker
    'docker': [r'^docker (build|compose|ps|logs|stop|start|restart) .+$', r'^docker compose (up|down|build|logs)( .+)?$'],

    # Database
    'psql': [r'^psql -f .+\.sql$', r'^psql -c ".+"$'],
    'mysql': [r'^mysql -e ".+"$'],
    'mongo': [r'^mongo .+$'],

    # Git
    'git': [r'^git (status|add|commit|push|pull|checkout|branch|log|diff|stash|reset)( .+)?$'],

    # Build tools
    'make': [r'^make( .+)?$'],
    'cmake': [r'^cmake .+$'],
    'cargo': [r'^cargo (build|run|test|check|clean)( .+)?$'],
    'go': [r'^go (build|run|test|mod .+)( .+)?$'],
}

class CommandExecutor:
    """Safely executes whitelisted commands with monitoring"""

    def __init__(self, process_monitor):
        self.process_monitor = process_monitor
        self.command_history = deque(maxlen=50)
        self.active_commands = {}  # command_id -> command_info

    def is_command_safe(self, command: str) -> tuple[bool, str]:
        """Check if command is in whitelist"""
        import re

        command = command.strip()

        # Extract the base command
        parts = command.split()
        if not parts:
            return False, "Empty command"

        base_cmd = parts[0]

        # Check if base command is in whitelist
        if base_cmd not in SAFE_COMMANDS:
            return False, f"Command '{base_cmd}' is not whitelisted"

        # Check if full command matches any pattern
        patterns = SAFE_COMMANDS[base_cmd]
        for pattern in patterns:
            if re.match(pattern, command):
                return True, "Command is safe"

        return False, f"Command '{command}' does not match whitelist patterns"

    def execute_command(self, command: str, cwd=None) -> dict:
        """Execute a safe command with monitoring"""
        import uuid

        # Check if command is safe
        is_safe, reason = self.is_command_safe(command)
        if not is_safe:
            return {
                'success': False,
                'error': reason,
                'command_id': None
            }

        # Generate command ID
        command_id = str(uuid.uuid4())[:8]

        # Start monitoring the command
        process, output_queue = self.process_monitor.monitor_command(command, cwd=cwd)

        if process is None:
            return {
                'success': False,
                'error': 'Failed to start process',
                'command_id': command_id
            }

        # Store command info
        self.active_commands[command_id] = {
            'command': command,
            'process': process,
            'output_queue': output_queue,
            'start_time': time.time(),
            'cwd': cwd
        }

        # Add to history
        self.command_history.append({
            'command_id': command_id,
            'command': command,
            'timestamp': datetime.now().isoformat(),
            'status': 'running'
        })

        return {
            'success': True,
            'command_id': command_id,
            'pid': process.pid
        }

    def get_command_output(self, command_id: str, max_lines=50) -> list:
        """Get output from a running command"""
        if command_id not in self.active_commands:
            return []

        output_queue = self.active_commands[command_id]['output_queue']
        output_lines = []

        try:
            while not output_queue.empty() and len(output_lines) < max_lines:
                stream_type, line = output_queue.get_nowait()
                output_lines.append({
                    'type': stream_type,
                    'line': line.strip()
                })
        except:
            pass

        return output_lines

    def get_command_status(self, command_id: str) -> dict:
        """Get status of a command"""
        if command_id not in self.active_commands:
            return {'status': 'unknown', 'exit_code': None}

        cmd_info = self.active_commands[command_id]
        process = cmd_info['process']
        exit_code = process.poll()

        if exit_code is None:
            return {'status': 'running', 'exit_code': None}
        else:
            # Update history
            for item in self.command_history:
                if item['command_id'] == command_id:
                    item['status'] = 'completed'
                    item['exit_code'] = exit_code

            return {'status': 'completed', 'exit_code': exit_code}

class GoalMemorySystem:
    """Remembers the user's original intent and tracks progress"""

    def __init__(self):
        self.current_goal = None
        self.sub_goals = []
        self.completed_tasks = []
        self.deviation_warnings = []
        self.start_time = None
        self.expected_duration = None  # in minutes

    def set_goal(self, goal_description: str, expected_duration: int = None):
        """Set the main goal before going to sleep"""
        self.current_goal = {
            'description': goal_description,
            'timestamp': datetime.now().isoformat(),
            'status': 'active'
        }
        self.start_time = time.time()
        self.expected_duration = expected_duration
        self.sub_goals = []
        self.completed_tasks = []
        self.deviation_warnings = []

        print(f"[GOAL] Set: {goal_description}")
        if expected_duration:
            print(f"[GOAL] Expected duration: {expected_duration} minutes")

    def add_sub_goal(self, sub_goal: str):
        """AI can break down goal into sub-tasks"""
        self.sub_goals.append({
            'description': sub_goal,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        })

    def mark_task_complete(self, task_description: str):
        """Mark a task as completed"""
        self.completed_tasks.append({
            'description': task_description,
            'timestamp': datetime.now().isoformat()
        })

    def add_deviation(self, deviation_description: str, severity: str = 'medium'):
        """Log when agent deviates from goal"""
        self.deviation_warnings.append({
            'description': deviation_description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        print(f"[DEVIATION WARNING] {severity.upper()}: {deviation_description}")

    def get_progress_summary(self) -> dict:
        """Get current progress towards goal"""
        elapsed = (time.time() - self.start_time) / 60 if self.start_time else 0

        return {
            'goal': self.current_goal,
            'sub_goals': self.sub_goals,
            'completed_tasks': self.completed_tasks,
            'deviation_warnings': self.deviation_warnings,
            'elapsed_minutes': round(elapsed, 1),
            'expected_minutes': self.expected_duration,
            'on_track': len(self.deviation_warnings) < 3
        }

class AgentOversightMonitor:
    """Dual-AI system that watches Claude Code agents"""

    def __init__(self, assistant):
        self.assistant = assistant
        self.goal_memory = GoalMemorySystem()
        self.primary_overseer = "groq-llama"  # Fast/cheap for continuous monitoring
        self.secondary_overseer = "claude-4-5-sonnet"  # Premium for decisions
        self.is_monitoring_agents = False
        self.last_screen_analysis = None
        self.suspicious_activity_count = 0

    def start_agent_oversight(self, user_goal: str, expected_duration: int = None):
        """Start monitoring Claude Code agents"""
        self.goal_memory.set_goal(user_goal, expected_duration)
        self.is_monitoring_agents = True
        self.suspicious_activity_count = 0
        print(f"[OVERSIGHT] Started monitoring agents for goal: {user_goal}")

    def stop_agent_oversight(self):
        """Stop monitoring"""
        self.is_monitoring_agents = False
        print("[OVERSIGHT] Stopped monitoring")

    def analyze_agent_activity(self, screenshot: str) -> dict:
        """Primary overseer (cheap AI) analyzes screen for agent activity"""
        if not self.is_monitoring_agents:
            return None

        model = AI_MODELS[self.primary_overseer]

        messages = [
            {
                "role": "system",
                "content": f"""You are monitoring a Claude Code agent to ensure it stays on task.

USER'S ORIGINAL GOAL: {self.goal_memory.current_goal['description'] if self.goal_memory.current_goal else 'None set'}

Your job:
1. Check if the agent is working on the user's goal
2. Detect if agent is:
   - Deleting large amounts of code unnecessarily
   - Skipping work or giving fake test results
   - Going off on tangents unrelated to the goal
   - Refusing to read files claiming they're "too big"
   - Creating unnecessary files instead of editing existing ones
3. Look for terminal output, file changes, test results

Respond with JSON:
{{
    "on_task": true/false,
    "current_activity": "brief description of what agent is doing",
    "suspicious": true/false,
    "concern_level": "none|low|medium|high|critical",
    "explanation": "why you're concerned or not",
    "recommendation": "what to do"
}}

Only respond with JSON, nothing else."""
            },
            {
                "role": "user",
                "content": "Analyze the current screen for agent activity."
            }
        ]

        try:
            # Use the cheap model for continuous monitoring
            response = self.assistant.get_ai_response(messages, screenshot if model.supports_vision else None)

            # Parse JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                self.last_screen_analysis = analysis

                # Track suspicious activity
                if analysis.get('suspicious'):
                    self.suspicious_activity_count += 1

                    # Escalate to premium AI if too many suspicions
                    if self.suspicious_activity_count >= 3:
                        return self._escalate_to_premium_overseer(screenshot, analysis)

                return analysis
        except Exception as e:
            print(f"[OVERSIGHT ERROR] {e}")

        return None

    def _escalate_to_premium_overseer(self, screenshot: str, primary_analysis: dict) -> dict:
        """Escalate to premium AI (Claude 4.5) for important decisions"""
        print(f"[OVERSIGHT] Escalating to premium overseer: {self.secondary_overseer}")

        model = AI_MODELS[self.secondary_overseer]

        messages = [
            {
                "role": "system",
                "content": f"""You are the SENIOR overseer AI. A junior AI has flagged suspicious activity.

USER'S GOAL: {self.goal_memory.current_goal['description'] if self.goal_memory.current_goal else 'None'}

JUNIOR AI'S ANALYSIS:
{json.dumps(primary_analysis, indent=2)}

COMPLETED TASKS SO FAR:
{json.dumps(self.goal_memory.completed_tasks[-5:], indent=2)}

PREVIOUS DEVIATIONS:
{json.dumps(self.goal_memory.deviation_warnings, indent=2)}

Your job:
1. Review the junior AI's concerns
2. Make a final decision
3. Decide if we should INTERVENE (pause agent, alert user, take over)

Respond with JSON:
{{
    "agree_with_junior": true/false,
    "severity": "none|low|medium|high|critical",
    "should_intervene": true/false,
    "intervention_type": "none|alert_user|pause_agent|take_control|redirect",
    "detailed_explanation": "full analysis of the situation",
    "suggested_action": "specific steps to take"
}}"""
            },
            {
                "role": "user",
                "content": "Review this situation and decide if we should intervene."
            }
        ]

        try:
            response = self.assistant.get_ai_response(messages, screenshot if model.supports_vision else None)

            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())

                # Log deviation if confirmed
                if decision.get('should_intervene'):
                    self.goal_memory.add_deviation(
                        decision.get('detailed_explanation', 'Unknown issue'),
                        severity=decision.get('severity', 'medium')
                    )

                # Reset suspicious count after escalation
                self.suspicious_activity_count = 0

                return {
                    **primary_analysis,
                    'escalated': True,
                    'senior_decision': decision
                }
        except Exception as e:
            print(f"[OVERSIGHT ESCALATION ERROR] {e}")

        return primary_analysis

class ProcessMonitor:
    """Monitors running processes and terminal output for errors"""

    def __init__(self):
        self.monitored_processes = {}  # pid -> process info
        self.output_queues = {}  # pid -> queue for output
        self.error_counts = {}  # command_type -> count
        self.last_error_time = {}  # command_type -> timestamp
        self.stuck_processes = {}  # pid -> start_time
        self.error_history = deque(maxlen=100)  # Keep last 100 errors
        self.is_monitoring = False

    def start_monitoring(self):
        """Start monitoring system processes"""
        self.is_monitoring = True
        print("[MONITOR] Process monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        print("[MONITOR] Process monitoring stopped")

    def get_active_window_process(self):
        """Get the currently active window process"""
        try:
            if sys.platform == 'win32':
                import win32gui
                import win32process
                hwnd = win32gui.GetForegroundWindow()
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                return psutil.Process(pid)
            else:
                # For Linux/Mac, would need different approach
                return None
        except Exception as e:
            print(f"[MONITOR] Error getting active window: {e}")
            return None

    def monitor_command(self, command, shell=True, cwd=None):
        """
        Start monitoring a shell command
        Returns: (process, output_queue)
        """
        try:
            # Create a queue for output
            output_queue = queue.Queue()

            # Start process
            process = subprocess.Popen(
                command,
                shell=shell,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=cwd,
                bufsize=1
            )

            # Store process info
            self.monitored_processes[process.pid] = {
                'process': process,
                'command': command,
                'start_time': time.time(),
                'output_queue': output_queue,
                'last_output_time': time.time(),
                'error_count': 0
            }

            # Start threads to read output
            def read_stdout():
                for line in iter(process.stdout.readline, ''):
                    if line:
                        output_queue.put(('stdout', line))
                        self._analyze_output(process.pid, line)
                        self.monitored_processes[process.pid]['last_output_time'] = time.time()

            def read_stderr():
                for line in iter(process.stderr.readline, ''):
                    if line:
                        output_queue.put(('stderr', line))
                        self._analyze_output(process.pid, line, is_error=True)
                        self.monitored_processes[process.pid]['last_output_time'] = time.time()

            threading.Thread(target=read_stdout, daemon=True).start()
            threading.Thread(target=read_stderr, daemon=True).start()

            print(f"[MONITOR] Started monitoring command: {command[:50]}...")
            return process, output_queue

        except Exception as e:
            print(f"[MONITOR] Error starting command monitoring: {e}")
            return None, None

    def _analyze_output(self, pid, line, is_error=False):
        """Analyze a line of output for errors"""
        import re

        if pid not in self.monitored_processes:
            return

        # Check for error patterns
        for pattern_name, pattern in ERROR_PATTERNS.items():
            if re.search(pattern, line, re.IGNORECASE):
                error_info = {
                    'pid': pid,
                    'command': self.monitored_processes[pid]['command'],
                    'pattern': pattern_name,
                    'line': line.strip(),
                    'timestamp': datetime.now().isoformat(),
                    'is_stderr': is_error
                }

                self.error_history.append(error_info)
                self.monitored_processes[pid]['error_count'] += 1

                # Update error counts by pattern type
                if pattern_name not in self.error_counts:
                    self.error_counts[pattern_name] = 0
                self.error_counts[pattern_name] += 1
                self.last_error_time[pattern_name] = time.time()

                print(f"[MONITOR] Error detected in {self.monitored_processes[pid]['command'][:30]}: {pattern_name}")

                return error_info

        return None

    def check_stuck_processes(self, timeout_seconds=300):
        """Check for processes that haven't produced output in a while"""
        stuck = []
        current_time = time.time()

        for pid, info in list(self.monitored_processes.items()):
            last_output = info.get('last_output_time', info['start_time'])
            elapsed = current_time - last_output

            # Check if process is still running
            try:
                process = info['process']
                if process.poll() is not None:
                    # Process finished
                    self.cleanup_process(pid)
                    continue
            except Exception:
                self.cleanup_process(pid)
                continue

            # Check if stuck
            if elapsed > timeout_seconds:
                stuck.append({
                    'pid': pid,
                    'command': info['command'],
                    'elapsed_seconds': elapsed,
                    'error_count': info['error_count']
                })

        return stuck

    def get_recent_errors(self, count=10):
        """Get recent errors"""
        return list(self.error_history)[-count:]

    def should_trigger_intervention(self):
        """
        Determine if AI should intervene based on error patterns
        Returns: (should_trigger, reason, details)
        """
        current_time = time.time()

        # Rule 1: 3+ consecutive errors of the same type within 2 minutes
        for pattern_name, count in self.error_counts.items():
            if count >= 3:
                last_error = self.last_error_time.get(pattern_name, 0)
                if current_time - last_error < 120:  # 2 minutes
                    return True, f"Multiple {pattern_name} errors", {
                        'pattern': pattern_name,
                        'count': count,
                        'recent_errors': [e for e in self.error_history if e['pattern'] == pattern_name][-3:]
                    }

        # Rule 2: Process stuck for >5 minutes
        stuck_processes = self.check_stuck_processes(timeout_seconds=300)
        if stuck_processes:
            return True, "Process appears stuck", stuck_processes[0]

        # Rule 3: High total error rate
        recent_errors = self.get_recent_errors(count=10)
        if len(recent_errors) >= 8:  # 8 errors in last 10 attempts
            return True, "High error rate detected", {
                'total_errors': len(recent_errors),
                'patterns': list(set([e['pattern'] for e in recent_errors]))
            }

        return False, None, None

    def cleanup_process(self, pid):
        """Clean up a finished process"""
        if pid in self.monitored_processes:
            info = self.monitored_processes[pid]
            exit_code = info['process'].poll()
            print(f"[MONITOR] Process {pid} finished with exit code: {exit_code}")
            del self.monitored_processes[pid]

    def get_status(self):
        """Get monitoring status"""
        return {
            'is_monitoring': self.is_monitoring,
            'active_processes': len(self.monitored_processes),
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts.copy(),
            'monitored_processes': [
                {
                    'pid': pid,
                    'command': info['command'][:50],
                    'error_count': info['error_count'],
                    'running_seconds': int(time.time() - info['start_time'])
                }
                for pid, info in self.monitored_processes.items()
            ]
        }

class EliteAIAssistant:
    def __init__(self):
        # Load API keys from environment variables
        self.api_keys = {
            "openai": os.getenv('OPENAI_API_KEY'),
            "anthropic": os.getenv('ANTHROPIC_API_KEY'),
            "openrouter": os.getenv('OPENROUTER_API_KEY'),
            "xai": os.getenv('XAI_API_KEY'),
            "groq": os.getenv('GROQ_API_KEY'),
            "together": os.getenv('TOGETHER_API_KEY'),
            "leonardo": os.getenv('LEONARDO_API_KEY')
        }
        self.clients = {}
        self.generated_floaters = []  # Cache for generated background floaters
        self.floater_generation_count = 0
        
        # Auto-initialize clients if keys are present
        if self.api_keys["openai"]:
            self.clients["openai"] = OpenAI(api_key=self.api_keys["openai"])
        if self.api_keys["anthropic"]:
            self.clients["anthropic"] = anthropic.Anthropic(api_key=self.api_keys["anthropic"])
        if self.api_keys["openrouter"]:
            self.clients["openrouter"] = {
                "api_key": self.api_keys["openrouter"],
                "base_url": "https://openrouter.ai/api/v1"
            }
        if self.api_keys["xai"]:
            self.clients["xai"] = OpenAI(
                api_key=self.api_keys["xai"],
                base_url="https://api.x.ai/v1"
            )
        if self.api_keys["groq"]:
            self.clients["groq"] = OpenAI(
                api_key=self.api_keys["groq"],
                base_url="https://api.groq.com/openai/v1"
            )
        if self.api_keys["together"]:
            self.clients["together"] = OpenAI(
                api_key=self.api_keys["together"],
                base_url="https://api.together.xyz/v1"
            )

        self.current_model = "claude-4-5-sonnet"
        self.is_watching = False
        self.is_controlling = False
        self.last_screenshot = None
        self.conversation_histories = {model_id: [] for model_id in AI_MODELS.keys()}
        self.screen_context = ""

        # Process monitoring
        self.process_monitor = ProcessMonitor()
        self.command_executor = CommandExecutor(self.process_monitor)
        self.guardian_mode = False  # Sleep mode for 6-8 hours

        # Agent Oversight System - AI babysitter for Claude Code
        self.agent_oversight = AgentOversightMonitor(self)

        # Initialize Leonardo AI client if key is present
        if self.api_keys["leonardo"]:
            self.leonardo_api_base = "https://cloud.leonardo.ai/api/rest/v1"
        
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a specific provider"""
        self.api_keys[provider] = api_key
        
        if provider == "openai":
            self.clients["openai"] = OpenAI(api_key=api_key)
        elif provider == "anthropic":
            self.clients["anthropic"] = anthropic.Anthropic(api_key=api_key)
        elif provider == "openrouter":
            self.clients["openrouter"] = {"api_key": api_key, "base_url": "https://openrouter.ai/api/v1"}
        elif provider == "xai":
            # xAI uses OpenAI-compatible API
            self.clients["xai"] = OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        
        return True
    
    def switch_model(self, model_id: str):
        """Switch to a different AI model"""
        if model_id in AI_MODELS:
            self.current_model = model_id
            return True
        return False
    
    def capture_screen(self) -> str:
        """Capture current screen and return as base64 string"""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Resize for faster processing
            img.thumbnail((1920, 1080), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            img.save(buffered, format="PNG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            self.last_screenshot = img_str
            return img_str
    
    def get_ai_response(self, messages: list, screenshot: str = None) -> str:
        """Get response from the current AI model"""
        model = AI_MODELS[self.current_model]
        provider = model.provider.value
        
        if self.api_keys[provider] is None:
            return f"Please set up your {provider.upper()} API key for {model.display_name}!"
        
        try:
            if provider == "openai":
                return self._get_openai_response(messages, screenshot, model.model_id)
            elif provider == "anthropic":
                return self._get_anthropic_response(messages, screenshot, model.model_id)
            elif provider == "openrouter":
                return self._get_openrouter_response(messages, screenshot, model.model_id)
            elif provider == "xai":
                return self._get_xai_response(messages, screenshot, model.model_id)
            elif provider == "groq":
                return self._get_groq_response(messages, screenshot, model.model_id)
            elif provider == "together":
                return self._get_together_response(messages, screenshot, model.model_id)
        except Exception as e:
            return f"Error with {model.display_name}: {str(e)}"
    
    def _get_openai_response(self, messages: list, screenshot: str, model_id: str) -> str:
        """Get response from OpenAI GPT-5"""
        client = self.clients["openai"]
        
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append(msg)
            else:
                content = []
                if isinstance(msg["content"], str):
                    content.append({"type": "text", "text": msg["content"]})
                else:
                    content = msg["content"]
                
                if screenshot and msg == messages[-1]:  # Add screenshot to last message
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot}",
                            "detail": "high"
                        }
                    })
                formatted_messages.append({"role": msg["role"], "content": content})
        
        response = client.chat.completions.create(
            model=model_id,
            messages=formatted_messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _get_anthropic_response(self, messages: list, screenshot: str, model_id: str) -> str:
        """Get response from Anthropic Claude 4.5 Sonnet"""
        client = self.clients["anthropic"]
        
        # Extract system message
        system_msg = ""
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                content = []
                if isinstance(msg["content"], str):
                    content.append({"type": "text", "text": msg["content"]})
                else:
                    content = msg["content"]
                
                if screenshot and msg == messages[-1]:  # Add screenshot to last message
                    content.insert(0, {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot
                        }
                    })
                formatted_messages.append({"role": msg["role"], "content": content})
        
        response = client.messages.create(
            model=model_id,
            max_tokens=1000,
            temperature=0.7,
            system=system_msg,
            messages=formatted_messages
        )
        
        return response.content[0].text
    
    def _get_openrouter_response(self, messages: list, screenshot: str, model_id: str) -> str:
        """Get response from OpenRouter (Claude 4 Opus)"""
        headers = {
            "Authorization": f"Bearer {self.api_keys['openrouter']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "AI Screen Assistant"
        }
        
        formatted_messages = []
        for msg in messages:
            content = []
            if isinstance(msg["content"], str):
                content.append({"type": "text", "text": msg["content"]})
            else:
                content = msg["content"]
            
            if screenshot and msg == messages[-1]:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{screenshot}"
                    }
                })
            formatted_messages.append({"role": msg["role"], "content": content})
        
        data = {
            "model": model_id,
            "messages": formatted_messages,
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"OpenRouter error: {response.text}"
    
    def _get_xai_response(self, messages: list, screenshot: str, model_id: str) -> str:
        """Get response from xAI Grok 4"""
        client = self.clients["xai"]
        
        formatted_messages = []
        for msg in messages:
            if msg["role"] == "system":
                formatted_messages.append(msg)
            else:
                content = []
                if isinstance(msg["content"], str):
                    content.append({"type": "text", "text": msg["content"]})
                else:
                    content = msg["content"]
                
                if screenshot and msg == messages[-1]:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot}",
                            "detail": "high"
                        }
                    })
                formatted_messages.append({"role": msg["role"], "content": content})
        
        response = client.chat.completions.create(
            model="grok-beta",  # Will be updated to grok-4 when available
            messages=formatted_messages,
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content

    def _get_groq_response(self, messages: list, screenshot: str, model_id: str) -> str:
        """Get response from Groq (Llama 3.1) - Ultra fast inference"""
        client = self.clients["groq"]

        # Groq doesn't support vision for Llama models, so we skip screenshot
        formatted_messages = []
        for msg in messages:
            if isinstance(msg["content"], str):
                formatted_messages.append(msg)
            else:
                # Extract only text content
                text_content = " ".join([c["text"] for c in msg["content"] if c.get("type") == "text"])
                formatted_messages.append({"role": msg["role"], "content": text_content})

        response = client.chat.completions.create(
            model=model_id,
            messages=formatted_messages,
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content

    def _get_together_response(self, messages: list, screenshot: str, model_id: str) -> str:
        """Get response from Together AI (Llama 3) - Fast and cheap"""
        client = self.clients["together"]

        # Together AI doesn't support vision for Llama models
        formatted_messages = []
        for msg in messages:
            if isinstance(msg["content"], str):
                formatted_messages.append(msg)
            else:
                # Extract only text content
                text_content = " ".join([c["text"] for c in msg["content"] if c.get("type") == "text"])
                formatted_messages.append({"role": msg["role"], "content": text_content})

        response = client.chat.completions.create(
            model=model_id,
            messages=formatted_messages,
            max_tokens=1000,
            temperature=0.7
        )

        return response.choices[0].message.content

    def analyze_screen(self, screenshot: str, user_message: str = None) -> str:
        """Analyze the screenshot and provide assistance"""
        model = AI_MODELS[self.current_model]
        
        messages = [
            {
                "role": "system",
                "content": f"""You are {model.display_name}, a helpful AI assistant watching someone's screen. 
                You can see what they're working on and help them when they're stuck.
                Be conversational and friendly - talk like a helpful colleague, not a robot.
                When you see they need help, offer specific suggestions.
                If they ask you to do something, explain what you're about to do in a natural way."""
            }
        ]
        
        # Add conversation history for context
        history = self.conversation_histories[self.current_model]
        for msg in history[-5:]:  # Last 5 messages
            messages.append(msg)
        
        # Current message
        content = user_message if user_message else "What's happening on the screen? Any suggestions?"
        messages.append({"role": "user", "content": content})
        
        return self.get_ai_response(messages, screenshot)
    
    def analyze_for_errors(self, screenshot: str) -> Optional[Dict[str, Any]]:
        """Analyze screenshot for errors, exceptions, and problems using AI vision"""
        model = AI_MODELS[self.current_model]
        
        # Only analyze if we have a valid API key and vision support
        if not self.api_keys[model.provider.value] or not model.supports_vision:
            return None
        
        messages = [
            {
                "role": "system",
                "content": """You are an expert error detection system. Analyze the screenshot for:
                - Red text indicating errors or exceptions
                - Error messages in terminals/consoles
                - Build failures
                - Dependency issues (npm, pip, etc.)
                - Syntax errors in code editors
                - Stack traces or tracebacks
                - Failed commands
                - Warning messages that need attention
                
                If you detect ANY problem, respond with JSON:
                {
                    "error_detected": true,
                    "type": "error_type",
                    "description": "brief description",
                    "severity": "critical|high|medium|low",
                    "suggested_fix": "what to do"
                }
                
                If no problems detected, respond with:
                {"error_detected": false}
                
                Be specific about what you see and where. Only respond with the JSON, nothing else."""
            },
            {
                "role": "user",
                "content": "Analyze this screenshot for errors or problems."
            }
        ]
        
        try:
            response = self.get_ai_response(messages, screenshot)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                if result.get("error_detected"):
                    return result
            return None
        except Exception as e:
            print(f"Error in analyze_for_errors: {e}")
            return None
    
    def get_control_instructions(self, task: str) -> list:
        """Get step-by-step instructions for controlling the computer"""
        screenshot = self.capture_screen()
        
        messages = [
            {
                "role": "system",
                "content": """You help control a computer by providing specific mouse and keyboard actions.
                Respond with a JSON array of actions like:
                [
                    {"action": "click", "x": 100, "y": 200},
                    {"action": "type", "text": "hello"},
                    {"action": "key", "keys": ["ctrl", "v"]},
                    {"action": "scroll", "direction": "down", "amount": 3}
                ]
                Be precise with coordinates based on the screenshot."""
            },
            {
                "role": "user",
                "content": f"Task: {task}"
            }
        ]
        
        response = self.get_ai_response(messages, screenshot)
        
        # Parse JSON response
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return []
    
    def execute_action(self, action: dict):
        """Execute a single control action"""
        try:
            if action["action"] == "click":
                pyautogui.click(action["x"], action["y"])
            elif action["action"] == "type":
                pyautogui.typewrite(action["text"], interval=0.02)
            elif action["action"] == "key":
                pyautogui.hotkey(*action["keys"])
            elif action["action"] == "scroll":
                amount = action.get("amount", 3)
                if action["direction"] == "down":
                    pyautogui.scroll(-amount * 100)
                else:
                    pyautogui.scroll(amount * 100)
            elif action["action"] == "move":
                pyautogui.moveTo(action["x"], action["y"], duration=0.3)
                
            time.sleep(0.3)  # Pause between actions
            
        except Exception as e:
            print(f"Error executing action: {e}")
    
    def generate_living_floater(self, interaction_type: str = "default") -> Optional[Dict[str, Any]]:
        """Generate a new background floater using Leonardo AI"""
        if not self.api_keys["leonardo"]:
            return None
        
        # Don't over-generate - limit to reasonable amount
        if self.floater_generation_count > 50:
            return None
        
        try:
            # Prompts based on interaction type
            prompts = {
                "default": "abstract cyberpunk geometric shape, hot pink neon glow, black void background, floating particle, minimalist, glowing edges, ethereal",
                "message": "abstract pink energy burst, cyberpunk particle effect, hot pink electric glow, black background, flowing wisps",
                "switch": "abstract tech symbol, hot pink holographic, black space, geometric patterns, glowing circuit",
                "watching": "abstract eye shape, pink neon outline, black void, surveillance aesthetic, glowing scan lines",
                "control": "abstract neural network node, hot pink synapses, black space, connected paths, glowing connections"
            }
            
            prompt = prompts.get(interaction_type, prompts["default"])
            
            # Leonardo AI API call to generate image
            headers = {
                "Authorization": f"Bearer {self.api_keys['leonardo']}",
                "Content-Type": "application/json"
            }
            
            # Request image generation
            generation_data = {
                "prompt": prompt,
                "negative_prompt": "text, words, letters, people, faces, realistic, photographic, green, blue, colorful",
                "modelId": "6bef9f1b-29cb-40c7-b9df-32b51c1f67d3",  # Leonardo Kino XL - good for abstract art
                "width": 512,
                "height": 512,
                "num_images": 1,
                "guidance_scale": 7,
                "public": False,
                "transparency": "foreground_only"
            }
            
            response = requests.post(
                f"{self.leonardo_api_base}/generations",
                headers=headers,
                json=generation_data,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"Leonardo API error: {response.text}")
                return None
            
            generation_id = response.json()["sdGenerationJob"]["generationId"]
            
            # Poll for completion (simplified - in production use webhooks)
            max_attempts = 30
            for attempt in range(max_attempts):
                time.sleep(2)  # Wait 2 seconds between checks
                
                status_response = requests.get(
                    f"{self.leonardo_api_base}/generations/{generation_id}",
                    headers=headers,
                    timeout=10
                )
                
                if status_response.status_code == 200:
                    generation_data = status_response.json()["generations_by_pk"]
                    
                    if generation_data["status"] == "COMPLETE":
                        image_url = generation_data["generated_images"][0]["url"]
                        
                        # Download and convert to base64
                        img_response = requests.get(image_url, timeout=30)
                        if img_response.status_code == 200:
                            img_base64 = base64.b64encode(img_response.content).decode()
                            
                            floater_data = {
                                "id": f"floater_{self.floater_generation_count}",
                                "image": img_base64,
                                "type": interaction_type,
                                "timestamp": datetime.now().isoformat()
                            }
                            
                            self.generated_floaters.append(floater_data)
                            self.floater_generation_count += 1
                            
                            return floater_data
                        break
            
            return None
            
        except Exception as e:
            print(f"Error generating floater: {e}")
            return None

# Create Flask app and SocketIO
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(32).hex())

# Dynamic CORS configuration to support any localhost port
CORS(app, origins=["http://127.0.0.1:*", "http://localhost:*"], supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global assistant instance
assistant = EliteAIAssistant()

# HTML Template with modern tabbed interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Elite AI Screen Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #000000;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            position: relative;
            overflow: hidden;
        }
        
        /* Starfield Canvas */
        #starfield {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
            pointer-events: none;
        }
        
        /* Dynamic Floater Container */
        #floater-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
            pointer-events: none;
            overflow: hidden;
        }
        
        /* AI-Generated Floater */
        .ai-floater {
            position: absolute;
            opacity: 0;
            animation: floaterFadeIn 1s ease-out forwards, floaterFloat 15s ease-in-out infinite;
            filter: drop-shadow(0 0 20px #FF1493);
            pointer-events: none;
        }
        
        @keyframes floaterFadeIn {
            from {
                opacity: 0;
                transform: scale(0.5);
            }
            to {
                opacity: 0.7;
                transform: scale(1);
            }
        }
        
        @keyframes floaterFloat {
            0%, 100% {
                transform: translate(0, 0) rotate(0deg);
            }
            25% {
                transform: translate(50px, -50px) rotate(90deg);
            }
            50% {
                transform: translate(0, -100px) rotate(180deg);
            }
            75% {
                transform: translate(-50px, -50px) rotate(270deg);
            }
        }
        
        /* Shooting Star Animation */
        .shooting-star {
            position: fixed;
            width: 2px;
            height: 2px;
            background: linear-gradient(90deg, #FF1493, transparent);
            border-radius: 50%;
            box-shadow: 0 0 20px #FF1493, 0 0 40px #FF1493;
            animation: shoot 3s linear infinite;
            z-index: 1;
        }
        
        @keyframes shoot {
            0% {
                transform: translate(0, 0) rotate(-45deg);
                opacity: 1;
                width: 2px;
                height: 100px;
            }
            70% {
                opacity: 1;
            }
            100% {
                transform: translate(1000px, 1000px) rotate(-45deg);
                opacity: 0;
            }
        }
        
        .container {
            background: #1a1a1a;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(255, 20, 147, 0.3);
            width: 100%;
            max-width: 1200px;
            height: 700px;
            display: flex;
            overflow: hidden;
            border: 1px solid #FF1493;
            position: relative;
            z-index: 10;
        }
        
        .sidebar {
            width: 350px;
            background: #0f0f0f;
            padding: 30px 20px;
            border-right: 1px solid #333;
            overflow-y: auto;
        }
        
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        h1 {
            color: #fff;
            font-size: 24px;
            margin-bottom: 30px;
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 800;
        }
        
        .ai-tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
            background: #1a1a1a;
            padding: 5px;
            border-radius: 12px;
        }
        
        .ai-tab {
            flex: 1;
            padding: 12px;
            background: transparent;
            border: none;
            color: #666;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s;
            font-weight: 600;
            font-size: 13px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }
        
        .ai-tab.active {
            background: linear-gradient(135deg, var(--tab-color) 0%, var(--tab-color-dark) 100%);
            color: white;
        }
        
        .ai-tab:hover:not(.active) {
            background: #2a2a2a;
            color: #999;
        }
        
        .ai-tab .icon {
            font-size: 20px;
        }
        
        .ai-indicator {
            height: 3px;
            background: var(--tab-color);
            margin-top: 5px;
            border-radius: 3px;
            width: 0;
            transition: width 0.3s;
        }
        
        .ai-tab.active .ai-indicator {
            width: 100%;
        }
        
        .status {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 25px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: #1a1a1a;
            border-radius: 10px;
            border: 1px solid #333;
        }
        
        .status-indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #333;
        }
        
        .status-indicator.active {
            background: #FF1493;
            box-shadow: 0 0 10px #FF1493;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 20, 147, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 20, 147, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 20, 147, 0); }
        }
        
        .api-status-display {
            margin-bottom: 25px;
            padding: 15px;
            background: #1a1a1a;
            border-radius: 10px;
            border: 1px solid #333;
        }
        
        .api-status-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            font-size: 13px;
        }
        
        .api-status {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #333;
            margin-left: 5px;
        }
        
        .api-status.connected {
            background: #FF1493;
            box-shadow: 0 0 5px #FF1493;
        }
        
        .controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        button {
            padding: 12px 20px;
            border: none;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            color: white;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #FF1493 0%, #C71585 100%);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 30px rgba(255, 20, 147, 0.6);
        }
        
        .btn-watch {
            background: linear-gradient(135deg, #FF1493 0%, #C71585 100%);
        }
        
        .btn-watch:hover {
            box-shadow: 0 5px 20px rgba(255, 20, 147, 0.5);
        }
        
        .btn-help {
            background: linear-gradient(135deg, #FF1493 0%, #C71585 100%);
        }
        
        .btn-help:hover {
            box-shadow: 0 5px 20px rgba(255, 20, 147, 0.5);
        }
        
        .btn-control {
            background: linear-gradient(135deg, #FF1493 0%, #C71585 100%);
        }
        
        .btn-control:hover {
            box-shadow: 0 5px 30px rgba(255, 20, 147, 0.6);
        }
        
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: #0f0f0f;
        }
        
        .chat-header {
            padding: 20px;
            background: #1a1a1a;
            border-bottom: 1px solid #333;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .chat-header h2 {
            color: #fff;
            font-size: 18px;
            font-weight: 600;
        }
        
        .current-ai-badge {
            padding: 6px 12px;
            background: linear-gradient(135deg, var(--current-color) 0%, var(--current-color-dark) 100%);
            border-radius: 20px;
            color: white;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            gap: 10px;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 16px;
            word-wrap: break-word;
            color: white;
        }
        
        .message.assistant .message-content {
            background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
            border: 1px solid #333;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, var(--current-color) 0%, var(--current-color-dark) 100%);
        }
        
        .message.system .message-content {
            background: #1a1a1a;
            border: 1px solid #ff9500;
            color: #ff9500;
            font-size: 13px;
        }
        
        .chat-input {
            display: flex;
            gap: 10px;
            padding: 20px;
            background: #1a1a1a;
            border-top: 1px solid #333;
        }
        
        .chat-input input {
            flex: 1;
            padding: 12px 20px;
            background: #0f0f0f;
            border: 1px solid #333;
            border-radius: 25px;
            color: white;
            font-size: 14px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .chat-input input:focus {
            border-color: var(--current-color);
        }
        
        .chat-input button {
            padding: 12px 30px;
            border-radius: 25px;
            background: linear-gradient(135deg, var(--current-color) 0%, var(--current-color-dark) 100%);
        }
        
        .screen-preview {
            width: 100%;
            height: 120px;
            background: #0f0f0f;
            border-radius: 10px;
            margin-top: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            font-size: 12px;
            overflow: hidden;
            border: 1px solid #333;
        }
        
        .screen-preview img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        :root {
            --tab-color: #FF1493;
            --tab-color-dark: #C71585;
            --current-color: #FF1493;
            --current-color-dark: #C71585;
        }
        
        /* Scrollbar styles */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0f0f0f;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Error Notification System */
        .notification-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
            max-width: 400px;
            pointer-events: none;
        }
        
        .error-notification {
            background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 100%);
            border-left: 4px solid #FF1493;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), 0 0 20px rgba(255, 20, 147, 0.3);
            display: flex;
            flex-direction: column;
            gap: 12px;
            animation: notificationSlideIn 0.4s ease-out;
            pointer-events: all;
            position: relative;
        }
        
        @keyframes notificationSlideIn {
            from {
                transform: translateX(450px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        .error-notification.critical {
            border-left-color: #dc2626;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), 0 0 30px rgba(220, 38, 38, 0.4);
        }
        
        .error-notification.high {
            border-left-color: #ff9500;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), 0 0 30px rgba(255, 149, 0, 0.4);
        }
        
        .error-notification.medium {
            border-left-color: #fbbf24;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), 0 0 30px rgba(251, 191, 36, 0.4);
        }
        
        .error-notification.low {
            border-left-color: #3b82f6;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.8), 0 0 30px rgba(59, 130, 246, 0.4);
        }
        
        .notification-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 10px;
        }
        
        .notification-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-weight: 700;
            font-size: 14px;
            color: #fff;
        }
        
        .notification-icon {
            font-size: 20px;
        }
        
        .notification-close {
            background: transparent;
            border: none;
            color: #666;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            transition: all 0.2s;
        }
        
        .notification-close:hover {
            background: #2a2a2a;
            color: #fff;
        }
        
        .notification-description {
            color: #ccc;
            font-size: 13px;
            line-height: 1.5;
        }
        
        .notification-fix {
            color: #FF1493;
            font-size: 12px;
            margin-top: 4px;
            font-weight: 600;
        }
        
        .notification-actions {
            display: flex;
            gap: 8px;
            margin-top: 8px;
        }
        
        .notification-btn {
            flex: 1;
            padding: 8px 16px;
            border: none;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .notification-btn-primary {
            background: linear-gradient(135deg, #FF1493 0%, #C71585 100%);
            color: white;
        }
        
        .notification-btn-primary:hover {
            box-shadow: 0 4px 15px rgba(255, 20, 147, 0.5);
            transform: translateY(-1px);
        }
        
        .notification-btn-secondary {
            background: #2a2a2a;
            color: #fff;
        }
        
        .notification-btn-secondary:hover {
            background: #333;
        }
        
        .notification-timestamp {
            color: #666;
            font-size: 11px;
            margin-top: 4px;
        }
        
        /* Typing Indicator */
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 10px;
            padding: 12px 16px;
            margin-bottom: 15px;
            animation: slideIn 0.3s ease;
        }
        
        .typing-indicator.show {
            display: flex;
        }
        
        .typing-indicator .message-content {
            background: linear-gradient(135deg, #2a2a2a 0%, #1a1a1a 100%);
            border: 1px solid #333;
            padding: 12px 16px;
            border-radius: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #FF1493;
            animation: typingDot 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) {
            animation-delay: 0.2s;
        }
        
        .typing-dot:nth-child(3) {
            animation-delay: 0.4s;
        }
        
        @keyframes typingDot {
            0%, 60%, 100% {
                transform: translateY(0);
                opacity: 0.7;
            }
            30% {
                transform: translateY(-10px);
                opacity: 1;
            }
        }
    </style>
</head>
<body>
    <!-- Starfield Background Canvas -->
    <canvas id="starfield"></canvas>
    
    <!-- AI-Generated Floater Container -->
    <div id="floater-container"></div>
    
    <!-- Error Notification Container -->
    <div class="notification-container" id="notification-container"></div>
    
    <div class="container">
        <div class="sidebar">
            <h1>🚀 Elite AI Assistant</h1>
            
            <div class="ai-tabs">
                <button class="ai-tab active" onclick="switchAI('gpt-5')" data-model="gpt-5" style="--tab-color: #00a67e; --tab-color-dark: #008866;">
                    <span class="icon">🧠</span>
                    <span>GPT-5</span>
                    <div class="ai-indicator"></div>
                </button>
                <button class="ai-tab" onclick="switchAI('claude-4-5-sonnet')" data-model="claude-4-5-sonnet" style="--tab-color: #d97706; --tab-color-dark: #b45309;">
                    <span class="icon">⚡</span>
                    <span>Claude 4.5</span>
                    <div class="ai-indicator"></div>
                </button>
                <button class="ai-tab" onclick="switchAI('claude-4-opus')" data-model="claude-4-opus" style="--tab-color: #7c3aed; --tab-color-dark: #6d28d9;">
                    <span class="icon">👑</span>
                    <span>Opus</span>
                    <div class="ai-indicator"></div>
                </button>
                <button class="ai-tab" onclick="switchAI('grok-4')" data-model="grok-4" style="--tab-color: #dc2626; --tab-color-dark: #b91c1c;">
                    <span class="icon">🚀</span>
                    <span>Grok 4</span>
                    <div class="ai-indicator"></div>
                </button>
            </div>
            
            <div class="status">
                <div class="status-item">
                    <div class="status-indicator" id="watching-status"></div>
                    <span style="color: #999;">Screen Watching</span>
                </div>
                <div class="status-item">
                    <div class="status-indicator" id="control-status"></div>
                    <span style="color: #999;">AI Control</span>
                </div>
                <div class="status-item">
                    <div class="status-indicator" id="monitor-status"></div>
                    <span style="color: #999;">Process Monitor</span>
                </div>
                <div class="status-item">
                    <div class="status-indicator" id="guardian-status"></div>
                    <span style="color: #999;">Guardian Mode</span>
                </div>
            </div>
            
            <div class="ai-homies-section">
                <h3 style="color: #FF1493; font-size: 14px; margin-bottom: 15px; text-transform: uppercase; letter-spacing: 1px; text-align: center; font-weight: 700;">Your AI Homies 🤖</h3>
                <div class="ai-homies-container">
                    <img src="/static/ai_homies.jpg" alt="AI Homies" style="width: 100%; border-radius: 15px; border: 2px solid #FF1493; box-shadow: 0 0 20px rgba(255, 20, 147, 0.5);">
                </div>
                <p style="color: #FF1493; font-size: 11px; margin-top: 12px; text-align: center; line-height: 1.4;">✨ All systems connected ✨</p>
            </div>
            
            <div class="controls">
                <button class="btn-watch" onclick="toggleWatching()">
                    <span id="watch-btn-text">Start Watching</span>
                </button>
                <button class="btn-help" onclick="requestHelp()">
                    Need Help
                </button>
                <button class="btn-monitor" onclick="toggleMonitoring()" style="background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);">
                    <span id="monitor-btn-text">Start Monitoring</span>
                </button>
                <button class="btn-guardian" onclick="toggleGuardian()" style="background: linear-gradient(135deg, #059669 0%, #047857 100%);">
                    <span id="guardian-btn-text">🌙 Guardian Mode</span>
                </button>
                <button class="btn-control" onclick="toggleControl()">
                    <span id="control-btn-text">Give AI Control</span>
                </button>
            </div>
            
            <div class="screen-preview" id="screen-preview">
                <span>Screen preview</span>
            </div>
        </div>
        
        <div class="main-content">
            <div class="chat-container">
                <div class="chat-header">
                    <h2>Chat with Elite AI</h2>
                    <div class="current-ai-badge" id="current-ai-badge">
                        <span class="icon">🧠</span>
                        <span>GPT-5</span>
                    </div>
                </div>
                
                <div class="chat-messages" id="chat-messages">
                    <div class="message system">
                        <div class="message-content">
                            Welcome to Elite AI Assistant! Connect your API keys to start using the most powerful AI models available. Switch between models anytime using the tabs.
                        </div>
                    </div>
                    
                    <!-- Typing Indicator -->
                    <div class="typing-indicator" id="typing-indicator">
                        <div class="message-content">
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                            <div class="typing-dot"></div>
                        </div>
                    </div>
                </div>

                <!-- PHASE 2: Command Execution Input -->
                <div class="chat-input" style="border-top: 2px solid #7c3aed; background: rgba(124, 58, 237, 0.1);">
                    <input type="text" id="command-input" placeholder="🚀 Run command (e.g., pip list, npm install package)..."
                           onkeypress="if(event.key==='Enter') runCommand()"
                           style="font-family: 'Courier New', monospace;">
                    <button onclick="runCommand()" style="background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);">
                        Run
                    </button>
                </div>

                <div class="chat-input">
                    <input type="text" id="message-input" placeholder="Type your message..."
                           onkeypress="if(event.key==='Enter') sendMessage()">
                    <button onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script>
        const socket = io();
        let isWatching = false;
        let isControlling = false;
        let isMonitoring = false;
        let isGuardian = false;
        let currentModel = 'gpt-5';
        let apiStatus = {
            openai: false,
            anthropic: false,
            openrouter: false,
            xai: false
        };
        
        const modelColors = {
            'gpt-5': { primary: '#00a67e', dark: '#008866' },
            'claude-4-5-sonnet': { primary: '#d97706', dark: '#b45309' },
            'claude-4-opus': { primary: '#7c3aed', dark: '#6d28d9' },
            'grok-4': { primary: '#dc2626', dark: '#b91c1c' }
        };
        
        const modelIcons = {
            'gpt-5': '🧠',
            'claude-4-5-sonnet': '⚡',
            'claude-4-opus': '👑',
            'grok-4': '🚀'
        };
        
        const modelNames = {
            'gpt-5': 'GPT-5',
            'claude-4-5-sonnet': 'Claude 4.5',
            'claude-4-opus': 'Opus',
            'grok-4': 'Grok 4'
        };
        
        socket.on('connect', () => {
            console.log('Connected to server');
            socket.emit('get_api_status');
        });
        
        socket.on('message', (data) => {
            hideTypingIndicator();
            addMessage(data.message, data.sender);
        });
        
        socket.on('new_floater', (data) => {
            addFloater(data);
        });
        
        // Handle autonomous error detection
        socket.on('error_detected', (data) => {
            console.log('[AUTONOMOUS] Error detected:', data);
            showErrorNotification(data);
        });
        
        socket.on('screen_update', (data) => {
            if (data.screenshot) {
                document.getElementById('screen-preview').innerHTML = 
                    `<img src="data:image/png;base64,${data.screenshot}" alt="Screen">`;
            }
        });
        
        socket.on('status_update', (data) => {
            if (data.watching !== undefined) {
                isWatching = data.watching;
                updateStatus('watching-status', data.watching);
                document.getElementById('watch-btn-text').textContent =
                    isWatching ? 'Stop Watching' : 'Start Watching';
            }
            if (data.controlling !== undefined) {
                isControlling = data.controlling;
                updateStatus('control-status', data.controlling);
                document.getElementById('control-btn-text').textContent =
                    isControlling ? 'Take Back Control' : 'Give AI Control';
            }
            if (data.monitoring !== undefined) {
                isMonitoring = data.monitoring;
                updateStatus('monitor-status', data.monitoring);
                document.getElementById('monitor-btn-text').textContent =
                    isMonitoring ? 'Stop Monitoring' : 'Start Monitoring';
            }
            if (data.guardian !== undefined) {
                isGuardian = data.guardian;
                updateStatus('guardian-status', data.guardian);
                document.getElementById('guardian-btn-text').textContent =
                    isGuardian ? '🌙 Guardian Active' : '🌙 Guardian Mode';
            }
            if (data.api_status) {
                updateAPIStatus(data.api_status);
            }
        });
        
        function updateStatus(elementId, active) {
            const element = document.getElementById(elementId);
            if (active) {
                element.classList.add('active');
            } else {
                element.classList.remove('active');
            }
        }
        
        function updateAPIStatus(status) {
            if (status) {
                apiStatus = status;
            }
            for (const [provider, connected] of Object.entries(apiStatus)) {
                const element = document.getElementById(`${provider}-status`);
                if (element) {
                    if (connected) {
                        element.classList.add('connected');
                    } else {
                        element.classList.remove('connected');
                    }
                }
            }
        }
        
        function switchAI(modelId) {
            currentModel = modelId;
            socket.emit('switch_model', {model_id: modelId});
            
            // Update UI
            document.querySelectorAll('.ai-tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelector(`[data-model="${modelId}"]`).classList.add('active');
            
            // Update colors
            const colors = modelColors[modelId];
            document.documentElement.style.setProperty('--current-color', colors.primary);
            document.documentElement.style.setProperty('--current-color-dark', colors.dark);
            
            // Update badge
            const badge = document.getElementById('current-ai-badge');
            badge.style.background = `linear-gradient(135deg, ${colors.primary} 0%, ${colors.dark} 100%)`;
            badge.innerHTML = `<span class="icon">${modelIcons[modelId]}</span><span>${modelNames[modelId]}</span>`;
            
            addMessage(`Switched to ${modelNames[modelId]}!`, 'system');
        }
        
        // Error Notification System
        let notificationId = 0;
        
        function showErrorNotification(errorData) {
            const container = document.getElementById('notification-container');
            const id = `notification-${notificationId++}`;
            
            const severityIcons = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🔵'
            };
            
            const severityTitles = {
                'critical': 'CRITICAL ERROR',
                'high': 'High Priority Error',
                'medium': 'Medium Priority Error',
                'low': 'Low Priority Issue'
            };
            
            const notification = document.createElement('div');
            notification.className = `error-notification ${errorData.severity}`;
            notification.id = id;
            
            notification.innerHTML = `
                <div class="notification-header">
                    <div class="notification-title">
                        <span class="notification-icon">${severityIcons[errorData.severity] || '⚠️'}</span>
                        <span>${severityTitles[errorData.severity] || 'Error Detected'}</span>
                    </div>
                    <button class="notification-close" onclick="dismissNotification('${id}')">×</button>
                </div>
                <div class="notification-description">
                    <strong>${errorData.type}:</strong> ${errorData.description}
                </div>
                <div class="notification-fix">
                    💡 ${errorData.suggested_fix}
                </div>
                <div class="notification-actions">
                    <button class="notification-btn notification-btn-primary" onclick="handleFixError('${id}', ${JSON.stringify(errorData).replace(/'/g, "&apos;")})">
                        Fix It Now
                    </button>
                    <button class="notification-btn notification-btn-secondary" onclick="dismissNotification('${id}')">
                        Dismiss
                    </button>
                </div>
                <div class="notification-timestamp">
                    Detected at ${new Date(errorData.timestamp).toLocaleTimeString()}
                </div>
            `;
            
            container.appendChild(notification);
            
            // Auto-dismiss low priority after 15 seconds
            if (errorData.severity === 'low') {
                setTimeout(() => dismissNotification(id), 15000);
            }
            
            // Play notification sound (optional)
            playNotificationSound();
        }
        
        function dismissNotification(id) {
            const notification = document.getElementById(id);
            if (notification) {
                notification.style.animation = 'notificationSlideIn 0.3s ease-out reverse';
                setTimeout(() => notification.remove(), 300);
            }
        }
        
        function handleFixError(notificationId, errorData) {
            console.log('[FIX] Attempting to fix error:', errorData);
            
            // Add message to chat
            addMessage(`🔧 Attempting to fix: ${errorData.description}`, 'system');
            
            // Emit fix request to backend (STEP 3)
            socket.emit('fix_error', {
                error: errorData,
                notification_id: notificationId
            });
            
            // Show feedback
            const notification = document.getElementById(notificationId);
            if (notification) {
                const actionsDiv = notification.querySelector('.notification-actions');
                actionsDiv.innerHTML = `
                    <button class="notification-btn notification-btn-primary" disabled style="opacity: 0.6;">
                        🔄 Fixing...
                    </button>
                `;
            }
        }
        
        function playNotificationSound() {
            // Simple notification sound using Web Audio API
            try {
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const oscillator = audioContext.createOscillator();
                const gainNode = audioContext.createGain();
                
                oscillator.connect(gainNode);
                gainNode.connect(audioContext.destination);
                
                oscillator.frequency.value = 800;
                oscillator.type = 'sine';
                
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);
                
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.2);
            } catch (e) {
                // Silently fail if audio not supported
            }
        }
        
        // Initialize API status on load
        socket.on('connect', () => {
            socket.emit('get_api_status');
        });
        
        socket.on('api_status', (status) => {
            updateAPIStatus(status);
        });
        
        function toggleWatching() {
            socket.emit('toggle_watching');
        }

        function toggleMonitoring() {
            socket.emit('toggle_monitoring');
        }

        function toggleGuardian() {
            if (!isGuardian) {
                if (confirm("Enable Guardian Mode? AI will monitor processes and intervene when errors are detected.")) {
                    socket.emit('toggle_guardian');
                    addMessage("🌙 Guardian Mode activated. I'll watch over your processes!", 'assistant');
                }
            } else {
                socket.emit('toggle_guardian');
                addMessage("Guardian Mode deactivated.", 'assistant');
            }
        }

        function toggleControl() {
            if (!isControlling) {
                if (confirm("Give the AI control of your mouse and keyboard?")) {
                    socket.emit('toggle_control');
                    addMessage("AI has control. Click 'Take Back Control' when ready!", 'assistant');
                }
            } else {
                socket.emit('toggle_control');
                addMessage("You've got control back!", 'assistant');
            }
        }
        
        function requestHelp() {
            socket.emit('request_help');
            addMessage("I need some help here!", 'user');
            showTypingIndicator();
        }
        
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();

            if (!message) return;

            addMessage(message, 'user');
            showTypingIndicator();
            socket.emit('chat_message', {message: message});
            input.value = '';
        }

        function runCommand() {
            const input = document.getElementById('command-input');
            const command = input.value.trim();

            if (!command) return;

            addMessage(`🚀 Running: ${command}`, 'user');
            socket.emit('execute_command', {command: command});
            input.value = '';
        }

        function showTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            indicator.classList.add('show');
            // Scroll to show the typing indicator
            indicator.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }
        
        function hideTypingIndicator() {
            const indicator = document.getElementById('typing-indicator');
            indicator.classList.remove('show');
        }
        
        function addMessage(message, sender) {
            const messagesDiv = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = message;

            messageDiv.appendChild(contentDiv);
            messagesDiv.appendChild(messageDiv);

            // Scroll to bottom - force scroll the container
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
    // Floater management
        let floaterCount = 0;
        const MAX_FLOATERS = 20;  // Limit to prevent performance issues
        
        // Initialize starfield background
        function initStarfield() {
            const canvas = document.getElementById('starfield');
            const ctx = canvas.getContext('2d');
            
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            
            const stars = [];
            for (let i = 0; i < 100; i++) {
                stars.push({
                    x: Math.random() * canvas.width,
                    y: Math.random() * canvas.height,
                    size: Math.random() * 2,
                    speed: Math.random() * 0.5
                });
            }
            
            function animateStars() {
                ctx.fillStyle = '#000000';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                stars.forEach(star => {
                    ctx.beginPath();
                    ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
                    ctx.fillStyle = '#FF1493';
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = '#FF1493';
                    ctx.fill();
                    
                    star.y += star.speed;
                    if (star.y > canvas.height) {
                        star.y = 0;
                        star.x = Math.random() * canvas.width;
                    }
                });
                
                requestAnimationFrame(animateStars);
            }
            
            animateStars();
            
            window.addEventListener('resize', () => {
                canvas.width = window.innerWidth;
                canvas.height = window.innerHeight;
            });
        }
        
        // Add AI-generated floater to the scene
        function addFloater(floaterData) {
            if (floaterCount >= MAX_FLOATERS) {
                // Remove oldest floater
                const container = document.getElementById('floater-container');
                const oldest = container.firstChild;
                if (oldest) {
                    oldest.remove();
                    floaterCount--;
                }
            }
            
            const container = document.getElementById('floater-container');
            const floater = document.createElement('img');
            floater.className = 'ai-floater';
            floater.src = `data:image/png;base64,${floaterData.image}`;
            floater.style.width = Math.random() * 100 + 50 + 'px';
            floater.style.height = 'auto';
            floater.style.left = Math.random() * (window.innerWidth - 150) + 'px';
            floater.style.top = Math.random() * (window.innerHeight - 150) + 'px';
            
            // Random animation duration
            floater.style.animationDuration = `${Math.random() * 10 + 15}s`;
            
            container.appendChild(floater);
            floaterCount++;
            
            console.log(`[LIVING UI] Generated ${floaterData.type} floater (#${floaterData.id})`);
            
            // Remove after animation completes a few cycles (to prevent memory buildup)
            setTimeout(() => {
                if (floater.parentNode) {
                    floater.remove();
                    floaterCount--;
                }
            }, 90000); // 90 seconds
        }
        
        // Initialize on page load
        initStarfield();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    # Disable caching to ensure latest version loads
    from flask import make_response
    response = make_response(render_template_string(HTML_TEMPLATE))
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        model_id: {
            "name": model.display_name,
            "provider": model.provider.value,
            "description": model.description,
            "supports_vision": model.supports_vision,
            "icon": model.icon
        }
        for model_id, model in AI_MODELS.items()
    })

@app.route('/api/generate_floater', methods=['POST'])
def generate_floater():
    """Generate a new background floater via API"""
    data = request.get_json()
    interaction_type = data.get('type', 'default')
    
    # Generate in background thread to avoid blocking
    def generate_and_emit():
        floater = assistant.generate_living_floater(interaction_type)
        if floater:
            socketio.emit('new_floater', floater, broadcast=True)
    
    threading.Thread(target=generate_and_emit, daemon=True).start()
    return jsonify({"status": "generating"})

@app.route('/api/floaters', methods=['GET'])
def get_floaters():
    """Get all cached floaters"""
    return jsonify({"floaters": assistant.generated_floaters})

@socketio.on('generate_floater')
def handle_generate_floater(data):
    """Handle floater generation request via WebSocket"""
    interaction_type = data.get('type', 'default')
    
    # Generate in background thread
    def generate_and_emit():
        floater = assistant.generate_living_floater(interaction_type)
        if floater:
            emit('new_floater', floater, broadcast=True)
    
    threading.Thread(target=generate_and_emit, daemon=True).start()

@socketio.on('get_api_status')
def handle_get_api_status():
    """Send current API key status to client"""
    api_status = {p: (k is not None) for p, k in assistant.api_keys.items()}
    emit('status_update', {'api_status': api_status})

@socketio.on('switch_model')
def handle_switch_model(data):
    model_id = data['model_id']
    if assistant.switch_model(model_id):
        # Generate a floater for model switch
        threading.Thread(
            target=lambda: assistant.generate_living_floater('switch'),
            daemon=True
        ).start()
        
        emit('message', {
            'message': f"Switched to {AI_MODELS[model_id].display_name}",
            'sender': 'system'
        }, broadcast=True)

@socketio.on('toggle_watching')
def handle_toggle_watching():
    assistant.is_watching = not assistant.is_watching
    emit('status_update', {'watching': assistant.is_watching}, broadcast=True)
    
    if assistant.is_watching:
        # Generate a floater for watching activation
        threading.Thread(
            target=lambda: assistant.generate_living_floater('watching'),
            daemon=True
        ).start()
        
        threading.Thread(target=watch_screen_loop, daemon=True).start()
        emit('message', {
            'message': "I'm watching your screen now. I'll help if you need it!",
            'sender': 'assistant'
        }, broadcast=True)
    else:
        emit('message', {
            'message': "Stopped watching. Let me know when you need me!",
            'sender': 'assistant'
        }, broadcast=True)

@socketio.on('toggle_control')
def handle_toggle_control():
    assistant.is_controlling = not assistant.is_controlling

    if assistant.is_controlling:
        # Generate a floater for control activation
        threading.Thread(
            target=lambda: assistant.generate_living_floater('control'),
            daemon=True
        ).start()

    emit('status_update', {'controlling': assistant.is_controlling}, broadcast=True)

@socketio.on('toggle_monitoring')
def handle_toggle_monitoring():
    """Toggle process monitoring"""
    if assistant.process_monitor.is_monitoring:
        assistant.process_monitor.stop_monitoring()
    else:
        assistant.process_monitor.start_monitoring()

    emit('status_update', {'monitoring': assistant.process_monitor.is_monitoring}, broadcast=True)
    emit('message', {
        'message': f"Process monitoring {'started' if assistant.process_monitor.is_monitoring else 'stopped'}!",
        'sender': 'system'
    }, broadcast=True)

@socketio.on('toggle_guardian')
def handle_toggle_guardian():
    """Toggle guardian mode (autonomous intervention)"""
    assistant.guardian_mode = not assistant.guardian_mode

    emit('status_update', {'guardian': assistant.guardian_mode}, broadcast=True)

    if assistant.guardian_mode:
        # Auto-enable monitoring and watching when guardian mode is on
        if not assistant.process_monitor.is_monitoring:
            assistant.process_monitor.start_monitoring()
            emit('status_update', {'monitoring': True}, broadcast=True)

        if not assistant.is_watching:
            assistant.is_watching = True
            emit('status_update', {'watching': True}, broadcast=True)
            threading.Thread(target=watch_screen_loop, daemon=True).start()

        # Start guardian loop
        threading.Thread(target=guardian_loop, daemon=True).start()

        emit('message', {
            'message': "🌙 Guardian Mode activated! I'm watching for errors and will intervene when needed.",
            'sender': 'assistant'
        }, broadcast=True)
    else:
        emit('message', {
            'message': "Guardian Mode deactivated.",
            'sender': 'system'
        }, broadcast=True)

@socketio.on('set_agent_goal')
def handle_set_agent_goal(data):
    """Set the goal for agent oversight - what you want done before you wake up"""
    goal = data.get('goal', '')
    expected_duration = data.get('expected_duration', None)  # in minutes

    if not goal:
        emit('message', {
            'message': "❌ Please provide a goal description",
            'sender': 'system'
        }, broadcast=True)
        return

    assistant.agent_oversight.start_agent_oversight(goal, expected_duration)

    emit('message', {
        'message': f"🎯 Goal set: {goal}\n{'Expected duration: ' + str(expected_duration) + ' minutes' if expected_duration else 'No time limit set'}\n\nI'll watch Claude Code agents and make sure they stay on task!",
        'sender': 'system'
    }, broadcast=True)

    emit('status_update', {
        'agent_oversight': True,
        'goal': goal
    }, broadcast=True)

@socketio.on('toggle_agent_oversight')
def handle_toggle_agent_oversight():
    """Toggle agent oversight monitoring"""
    if assistant.agent_oversight.is_monitoring_agents:
        assistant.agent_oversight.stop_agent_oversight()

        emit('message', {
            'message': "Agent oversight stopped.",
            'sender': 'system'
        }, broadcast=True)
    else:
        emit('message', {
            'message': "⚠️ Please set a goal first using 'Set Agent Goal'",
            'sender': 'system'
        }, broadcast=True)

    emit('status_update', {
        'agent_oversight': assistant.agent_oversight.is_monitoring_agents
    }, broadcast=True)

@socketio.on('get_agent_progress')
def handle_get_agent_progress():
    """Get current progress towards the goal"""
    progress = assistant.agent_oversight.goal_memory.get_progress_summary()

    emit('agent_progress', progress, broadcast=True)

    # Send formatted message to chat
    if progress['goal']:
        status_msg = f"""📊 **Agent Oversight Progress Report**

**Goal:** {progress['goal']['description']}
**Time Elapsed:** {progress['elapsed_minutes']} minutes
{'**Expected:** ' + str(progress['expected_minutes']) + ' minutes' if progress['expected_minutes'] else ''}

**Completed Tasks:** {len(progress['completed_tasks'])}
**Warnings:** {len(progress['deviation_warnings'])}

**Status:** {'✅ On Track' if progress['on_track'] else '⚠️ Issues Detected'}
"""

        if progress['deviation_warnings']:
            status_msg += "\n**Recent Warnings:**\n"
            for warning in progress['deviation_warnings'][-3:]:
                status_msg += f"- [{warning['severity'].upper()}] {warning['description']}\n"

        emit('message', {
            'message': status_msg,
            'sender': 'system'
        }, broadcast=True)
    else:
        emit('message', {
            'message': "No active oversight goal set.",
            'sender': 'system'
        }, broadcast=True)

@socketio.on('request_help')
def handle_help_request():
    model = AI_MODELS[assistant.current_model]
    if assistant.api_keys[model.provider.value]:
        screenshot = assistant.capture_screen()
        response = assistant.analyze_screen(screenshot, "The user needs help with what's on screen")
        emit('message', {'message': response, 'sender': 'assistant'}, broadcast=True)
    else:
        emit('message', {
            'message': f"Please set up the API key for {model.display_name}!",
            'sender': 'system'
        }, broadcast=True)

@socketio.on('chat_message')
def handle_chat_message(data):
    message = data['message']
    model = AI_MODELS[assistant.current_model]
    
    # Generate a floater for this interaction
    threading.Thread(
        target=lambda: assistant.generate_living_floater('message'),
        daemon=True
    ).start()
    
    # Add to conversation history
    assistant.conversation_histories[assistant.current_model].append({
        "role": "user",
        "content": message
    })
    
    if assistant.api_keys[model.provider.value]:
        # Get response based on screen context if watching
        if assistant.is_watching and assistant.last_screenshot and model.supports_vision:
            response = assistant.analyze_screen(assistant.last_screenshot, message)
        else:
            # Regular chat without screen context
            messages = [
                {
                    "role": "system",
                    "content": f"You are {model.display_name}, a friendly and helpful AI assistant. Be conversational and natural."
                },
                *assistant.conversation_histories[assistant.current_model][-10:]
            ]
            response = assistant.get_ai_response(messages)
        
        assistant.conversation_histories[assistant.current_model].append({
            "role": "assistant",
            "content": response
        })
        emit('message', {'message': response, 'sender': 'assistant'}, broadcast=True)
        
        # Check if user wants AI to do something
        action_keywords = ['do it', 'take control', 'handle it', 'fix it', 'help me with']
        if any(keyword in message.lower() for keyword in action_keywords) and not assistant.is_controlling:
            emit('message', {
                'message': "Want me to take control? Click 'Give AI Control'!",
                'sender': 'assistant'
            }, broadcast=True)
    else:
        emit('message', {
            'message': f"Please set up the API key for {model.display_name}!",
            'sender': 'system'
        }, broadcast=True)

def watch_screen_loop():
    """Background thread to watch screen with autonomous error detection"""
    while assistant.is_watching:
        try:
            screenshot = assistant.capture_screen()
            socketio.emit('screen_update', {'screenshot': screenshot})

            # STEP 1: Autonomous error detection
            error_detected = assistant.analyze_for_errors(screenshot)

            if error_detected:
                print(f"[AUTONOMOUS] Error detected: {error_detected['type']} - {error_detected['description']}")

                # Emit error detection event to UI
                socketio.emit('error_detected', {
                    'type': error_detected.get('type', 'unknown'),
                    'description': error_detected.get('description', 'Unknown error'),
                    'severity': error_detected.get('severity', 'medium'),
                    'suggested_fix': error_detected.get('suggested_fix', 'No suggestion available'),
                    'timestamp': datetime.now().isoformat()
                })

            # STEP 2: Agent Oversight - Watch Claude Code agents
            if assistant.agent_oversight.is_monitoring_agents:
                agent_analysis = assistant.agent_oversight.analyze_agent_activity(screenshot)

                if agent_analysis:
                    # Emit agent activity to UI
                    socketio.emit('agent_activity', agent_analysis)

                    # Check if we need to intervene
                    if agent_analysis.get('escalated') and agent_analysis.get('senior_decision', {}).get('should_intervene'):
                        socketio.emit('agent_intervention_required', {
                            'analysis': agent_analysis,
                            'timestamp': datetime.now().isoformat()
                        })

                        socketio.emit('message', {
                            'message': f"🚨 AGENT OVERSIGHT: {agent_analysis['senior_decision']['detailed_explanation']}",
                            'sender': 'system'
                        }, broadcast=True)

            # If AI has control, check for tasks
            if assistant.is_controlling:
                # This would be more sophisticated in production
                pass

            time.sleep(3)  # Update every 3 seconds
        except Exception as e:
            print(f"Error in watch loop: {e}")
            time.sleep(5)

def guardian_loop():
    """Background thread for guardian mode - monitors processes and intervenes when needed"""
    print("[GUARDIAN] Guardian loop started")

    while assistant.guardian_mode:
        try:
            # Check if we should trigger intervention
            should_trigger, reason, details = assistant.process_monitor.should_trigger_intervention()

            if should_trigger:
                print(f"[GUARDIAN] Intervention triggered: {reason}")
                print(f"[GUARDIAN] Details: {details}")

                # Notify user
                socketio.emit('message', {
                    'message': f"🚨 Guardian detected: {reason}",
                    'sender': 'system'
                }, broadcast=True)

                # Get recent errors for context
                recent_errors = assistant.process_monitor.get_recent_errors(count=5)

                # Create error summary for AI
                error_summary = f"""Guardian Mode Alert:
Reason: {reason}

Recent Errors:
"""
                for err in recent_errors:
                    error_summary += f"- [{err['pattern']}] {err['line'][:100]}\n"

                error_summary += f"\nDetails: {json.dumps(details, indent=2)}"

                # Ask AI for intervention plan
                socketio.emit('message', {
                    'message': "🤖 Analyzing the situation and preparing intervention...",
                    'sender': 'assistant'
                }, broadcast=True)

                # Capture screen for context
                screenshot = assistant.capture_screen()

                # Get AI response with context
                model = AI_MODELS[assistant.current_model]
                messages = [
                    {
                        "role": "system",
                        "content": """You are a Guardian AI monitoring a developer's workspace. You detect errors and provide actionable solutions.
When you see errors, provide:
1. Clear explanation of what's wrong
2. Step-by-step fix instructions
3. Commands to run (if applicable)

Be concise and actionable."""
                    },
                    {
                        "role": "user",
                        "content": error_summary
                    }
                ]

                response = assistant.get_ai_response(messages, screenshot if model.supports_vision else None)

                socketio.emit('message', {
                    'message': response,
                    'sender': 'assistant'
                }, broadcast=True)

                # Reset error counts after intervention
                assistant.process_monitor.error_counts.clear()

                # Wait before checking again to avoid spam
                time.sleep(30)
            else:
                # Check every 10 seconds when no intervention needed
                time.sleep(10)

        except Exception as e:
            print(f"[GUARDIAN] Error in guardian loop: {e}")
            time.sleep(10)

    print("[GUARDIAN] Guardian loop stopped")

@socketio.on('execute_command')
def handle_execute_command(data):
    """Execute a safe command with monitoring - PHASE 2"""
    command = data.get('command', '').strip()
    cwd = data.get('cwd', None)

    if not command:
        emit('command_error', {'error': 'No command provided'})
        return

    print(f"[COMMAND] Executing: {command}")

    # Execute command
    result = assistant.command_executor.execute_command(command, cwd=cwd)

    if result['success']:
        emit('command_started', {
            'command_id': result['command_id'],
            'command': command,
            'pid': result['pid']
        }, broadcast=True)

        emit('message', {
            'message': f"🚀 Running: {command}",
            'sender': 'system'
        }, broadcast=True)

        # Start a thread to monitor output
        def stream_output():
            command_id = result['command_id']
            while True:
                status = assistant.command_executor.get_command_status(command_id)
                output = assistant.command_executor.get_command_output(command_id)

                if output:
                    socketio.emit('command_output', {
                        'command_id': command_id,
                        'output': output
                    })

                if status['status'] == 'completed':
                    exit_code = status['exit_code']
                    success_msg = f"✅ Command completed (exit code: {exit_code})" if exit_code == 0 else f"❌ Command failed (exit code: {exit_code})"

                    socketio.emit('command_completed', {
                        'command_id': command_id,
                        'exit_code': exit_code,
                        'success': exit_code == 0
                    })

                    socketio.emit('message', {
                        'message': success_msg,
                        'sender': 'system'
                    })
                    break

                time.sleep(0.5)

        threading.Thread(target=stream_output, daemon=True).start()

    else:
        emit('command_error', {
            'error': result['error'],
            'command': command
        }, broadcast=True)

        emit('message', {
            'message': f"❌ Command blocked: {result['error']}",
            'sender': 'system'
        }, broadcast=True)

@socketio.on('get_command_history')
def handle_get_command_history():
    """Get command execution history"""
    history = list(assistant.command_executor.command_history)
    emit('command_history', {'history': history})

@socketio.on('fix_error')
def handle_fix_error(data):
    """Handle one-click error fixing - PHASE 2: Command Execution"""
    error_data = data.get('error', {})
    notification_id = data.get('notification_id', '')

    print(f"[FIX] Received fix request for: {error_data.get('type', 'unknown')}")

    model = AI_MODELS[assistant.current_model]
    if not assistant.api_keys[model.provider.value]:
        emit('message', {
            'message': f"Cannot fix error: {model.display_name} API key not set",
            'sender': 'system'
        }, broadcast=True)
        return

    try:
        # Construct task for AI to provide COMMAND-BASED fix
        task = f"""Fix this error by providing a shell command:
Type: {error_data.get('type', 'Unknown')}
Description: {error_data.get('description', 'No description')}
Suggested fix: {error_data.get('suggested_fix', 'No suggestion')}
Severity: {error_data.get('severity', 'medium')}

Provide ONE shell command to fix this error. The command must be safe (npm, pip, git, docker, etc.).
ONLY respond with the command, nothing else. For example: "npm install missing-package"
"""
        
        emit('message', {
            'message': f"🔧 Asking AI for fix command...",
            'sender': 'system'
        }, broadcast=True)

        # Get AI response with command
        messages = [{"role": "user", "content": task}]
        ai_response = assistant.get_ai_response(messages)

        if not ai_response:
            emit('message', {
                'message': "❌ Could not get fix command from AI.",
                'sender': 'system'
            }, broadcast=True)
            return

        # Extract command from response (remove quotes, trim whitespace)
        command = ai_response.strip().strip('"').strip("'").strip()

        emit('message', {
            'message': f"💡 Suggested fix: {command}",
            'sender': 'assistant'
        }, broadcast=True)

        # Execute the command
        result = assistant.command_executor.execute_command(command)

        if result['success']:
            emit('message', {
                'message': f"🚀 Executing fix command...",
                'sender': 'system'
            }, broadcast=True)

            # Start streaming output
            def stream_fix_output():
                command_id = result['command_id']
                while True:
                    status = assistant.command_executor.get_command_status(command_id)
                    output = assistant.command_executor.get_command_output(command_id)

                    if output:
                        for line in output:
                            socketio.emit('message', {
                                'message': f"  {line['line']}",
                                'sender': 'system'
                            })

                    if status['status'] == 'completed':
                        exit_code = status['exit_code']
                        if exit_code == 0:
                            socketio.emit('message', {
                                'message': f"✅ Fix applied successfully!",
                                'sender': 'system'
                            })
                        else:
                            socketio.emit('message', {
                                'message': f"❌ Fix failed (exit code: {exit_code})",
                                'sender': 'system'
                            })
                        break

                    time.sleep(0.5)

            threading.Thread(target=stream_fix_output, daemon=True).start()

        else:
            emit('message', {
                'message': f"❌ Cannot execute command: {result['error']}",
                'sender': 'system'
            }, broadcast=True)
        
        emit('message', {
            'message': "✅ Fix sequence completed! Check if the error is resolved.",
            'sender': 'assistant'
        }, broadcast=True)
        
        # Emit fix completion event
        emit('fix_completed', {
            'notification_id': notification_id,
            'success': True
        }, broadcast=True)
        
    except Exception as e:
        print(f"[FIX ERROR] Exception: {e}")
        emit('message', {
            'message': f"❌ Error during fix attempt: {str(e)}",
            'sender': 'system'
        }, broadcast=True)
        
        emit('fix_completed', {
            'notification_id': notification_id,
            'success': False,
            'error': str(e)
        }, broadcast=True)


if __name__ == '__main__':
    import sys
    # Set UTF-8 encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    host = os.getenv('HOST', '127.0.0.1')  # Localhost only by default!
    port = int(os.getenv('PORT', 5000))
    
    print("=" * 60)
    print(">> Elite AI Screen Assistant Starting...")
    print("=" * 60)
    print(f"[*] Server: {host}:{port} (localhost only)")
    print(f"[*] API Keys loaded from .env file")
    print("")
    print("[+] Loaded API Keys:")
    
    keys_found = False
    for provider, key in assistant.api_keys.items():
        if key:
            # Show first 8 and last 4 characters for verification
            masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
            print(f"    [OK] {provider.upper()}: {masked_key}")
            keys_found = True
        else:
            print(f"    [ ] {provider.upper()}: Not set")
    
    if not keys_found:
        print("")
        print("[!] WARNING: No API keys found in .env file!")
        print("    Create a .env file with your API keys to use the assistant.")
    
    print("")
    print("=" * 60)
    print(f"[>] Open your browser: http://{host}:{port}")
    print("=" * 60)
    print("")
    print("[!] SECURITY NOTES:")
    print("    - Server only accessible from localhost")
    print("    - Do not expose this to the internet")
    print("    - Monitor your API usage and costs")
    print("")
    
    socketio.run(app, debug=False, port=port, host=host)
