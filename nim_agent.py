#!/usr/bin/env python3
"""
NIM Agent - Interactive Terminal Agent with Auto Stage Detection, Goals, Personas & File Tools
Each project has TWO model options: GLM-5.2 and Nemotron Ultra.
Auto-detects development stage and applies the right persona automatically.
Uses current working directory as the project directory.
Preserves context across relaunches. API keys persisted after first setup.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    import urllib.request

# ============================================
# PATHS
# ============================================

CONFIG_FILE = Path.home() / ".nim_agent_config.json"
SESSIONS_FILE = Path.home() / ".nim_agent_sessions.json"
HISTORY_FILE = Path.home() / ".nim_agent_history.json"
NIM_BASE = "https://integrate.api.nvidia.com/v1"

# ============================================
# DEFAULT PROFILE TEMPLATES (no keys, no project paths)
# ============================================

PROFILE_TEMPLATES = {
    "1": {
        "name": "OMEGA",
        "models": {
            "glm": {"name": "OMEGA-GLM5", "id": "z-ai/glm-5.2"},
            "nemotron": {"name": "OMEGA-Nemotron", "id": "nvidia/nemotron-3-ultra-550b-a55b"},
        },
        "default_model": "glm",
    },
    "2": {
        "name": "Antigravity",
        "models": {
            "glm": {"name": "Antigravity-GLM5", "id": "z-ai/glm-5.2"},
            "nemotron": {"name": "Antigravity-Nemotron", "id": "nvidia/nemotron-3-ultra-550b-a55b"},
        },
        "default_model": "nemotron",
    },
    "3": {
        "name": "CryptoTrader",
        "models": {
            "glm": {"name": "CryptoTrader-GLM5", "id": "z-ai/glm-5.2"},
            "nemotron": {"name": "CryptoTrader-Nemotron", "id": "nvidia/nemotron-3-ultra-550b-a55b"},
        },
        "default_model": "glm",
    },
    "4": {
        "name": "ASES",
        "models": {
            "glm": {"name": "ASES-GLM5", "id": "z-ai/glm-5.2"},
            "nemotron": {"name": "ASES-Nemotron", "id": "nvidia/nemotron-3-ultra-550b-a55b"},
        },
        "default_model": "nemotron",
    },
}

# ============================================
# CONFIG MANAGEMENT
# ============================================

def load_config() -> Dict:
    """Load config from disk or return empty dict."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_config(config: Dict):
    """Save config to disk."""
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

def prompt_for_keys() -> Dict:
    """Interactive first-time setup for API keys."""
    print("=" * 60)
    print("  FIRST-TIME SETUP: Enter your NVIDIA API keys")
    print("  (Leave blank to skip a profile)")
    print("=" * 60)
    config = {"profiles": {}}
    for key, tmpl in PROFILE_TEMPLATES.items():
        val = input(f"  API key for [{key}] {tmpl['name']}: ").strip()
        if val:
            config["profiles"][key] = {
                "name": tmpl["name"],
                "key": val,
                "models": tmpl["models"],
                "default_model": tmpl["default_model"],
            }
    if not config["profiles"]:
        print("  [WARNING] No keys provided. You can add them later with /key")
    save_config(config)
    print(f"  [Saved] API keys stored in {CONFIG_FILE}\n")
    return config

def get_profiles() -> Dict:
    """Get profiles, prompting for keys on first run."""
    config = load_config()
    if not config.get("profiles"):
        config = prompt_for_keys()
    return config["profiles"]

def update_key(profile_id: str, new_key: str):
    """Update a single API key in config."""
    config = load_config()
    if "profiles" not in config:
        config["profiles"] = {}
    if profile_id in PROFILE_TEMPLATES:
        tmpl = PROFILE_TEMPLATES[profile_id]
        config["profiles"][profile_id] = {
            "name": tmpl["name"],
            "key": new_key,
            "models": tmpl["models"],
            "default_model": tmpl["default_model"],
        }
        save_config(config)
        return True
    return False

# ============================================
# SESSION MANAGEMENT (context persistence)
# ============================================

def load_sessions() -> Dict:
    """Load saved sessions from disk."""
    if SESSIONS_FILE.exists():
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_sessions(sessions: Dict):
    """Save sessions to disk."""
    with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)

def get_session_key(profile_name: str, project_dir: str) -> str:
    """Unique key for a profile+project session."""
    safe_dir = project_dir.replace("\\", "/").replace(":", "_")
    return f"{profile_name}::{safe_dir}"

def load_session(profile_name: str, project_dir: str) -> Optional[Dict]:
    """Load a saved session if it exists."""
    sessions = load_sessions()
    key = get_session_key(profile_name, project_dir)
    return sessions.get(key)

def save_session(profile_name: str, project_dir: str, data: Dict):
    """Save current session state."""
    sessions = load_sessions()
    key = get_session_key(profile_name, project_dir)
    sessions[key] = data
    # Keep only last 20 sessions to avoid bloat
    if len(sessions) > 20:
        oldest = sorted(sessions.items(), key=lambda x: x[1].get("timestamp", ""))[:len(sessions)-20]
        for k, _ in oldest:
            del sessions[k]
    save_sessions(sessions)

def delete_session(profile_name: str, project_dir: str):
    """Delete a saved session."""
    sessions = load_sessions()
    key = get_session_key(profile_name, project_dir)
    if key in sessions:
        del sessions[key]
        save_sessions(sessions)

# ============================================
# AGENT PERSONAS
# ============================================

PERSONAS = {
    "architect": {
        "name": "Architect",
        "description": "Design systems and review architecture",
        "stage": "planning",
        "prompt": "You are a senior software architect. Analyze the project structure, suggest improvements, design patterns, and scalability solutions. Think about maintainability, performance, and clean separation of concerns. Design before building."
    },
    "coder": {
        "name": "Coder",
        "description": "Write, refactor, and debug code",
        "stage": "implementation",
        "prompt": "You are an expert software engineer. Write clean, efficient, well-documented code. Follow best practices and modern conventions. Always explain your reasoning. Focus on completing features and fixing TODOs."
    },
    "debugger": {
        "name": "Debugger",
        "description": "Find and fix bugs",
        "stage": "troubleshooting",
        "prompt": "You are a debugging specialist. Analyze error messages, trace logic, identify root causes, and provide minimal fixes. Focus on correctness. Hunt down bugs and edge cases."
    },
    "reviewer": {
        "name": "Reviewer",
        "description": "Code review and quality analysis",
        "stage": "review",
        "prompt": "You are a strict code reviewer. Find bugs, security issues, performance bottlenecks, and style violations. Be thorough and critical. Suggest specific fixes. Ensure code quality before merge."
    },
    "optimizer": {
        "name": "Optimizer",
        "description": "Performance optimization",
        "stage": "optimization",
        "prompt": "You are a performance engineer. Profile code, identify bottlenecks, suggest algorithmic improvements, and optimize for speed and memory. Make fast code faster."
    },
    "general": {
        "name": "General",
        "description": "General purpose assistant",
        "stage": "general",
        "prompt": "You are a helpful AI assistant. Answer questions, explain concepts, and assist with tasks."
    },
}

# ============================================
# STAGE DETECTOR
# ============================================

class StageDetector:
    def __init__(self, file_tools: "FileTools"):
        self.tools = file_tools
        self.base = file_tools.base

    def detect(self) -> Tuple[str, str]:
        checks = [
            ("debugger", self._check_debugger_signals),
            ("optimizer", self._check_optimizer_signals),
            ("reviewer", self._check_reviewer_signals),
            ("coder", self._check_coder_signals),
            ("architect", self._check_architect_signals),
        ]
        for persona_key, check_fn in checks:
            result = check_fn()
            if result:
                return persona_key, result
        return "general", "No specific stage detected. Using general assistant."

    def _check_debugger_signals(self) -> Optional[str]:
        error_patterns = [
            r"error\.log", r"debug\.log", r"traceback", r"crash",
            r"\.err$", r"failures\.txt", r"pytest_cache"
        ]
        error_files = []
        try:
            for root, _, files in os.walk(self.base):
                for f in files:
                    if any(re.search(p, f, re.I) for p in error_patterns):
                        error_files.append(f)
                    elif f.endswith(".log"):
                        fpath = Path(root) / f
                        try:
                            with open(fpath, "r", errors="replace") as fh:
                                content = fh.read(2000)
                                if any(sig in content.lower() for sig in ["traceback", "exception", "error:", "failed"]):
                                    error_files.append(f)
                        except:
                            pass
                if error_files:
                    break
        except:
            pass
        if error_files:
            return f"Found error signals: {', '.join(error_files[:3])}. Switching to Debugger."
        try:
            result = self.tools.search(r"BUG|HACK|FIXME|XXX", ".")
            if not result.startswith("[INFO]") and result.count(":") > 3:
                return f"Found {result.count(':')} bug/hack markers. Switching to Debugger."
        except:
            pass
        return None

    def _check_optimizer_signals(self) -> Optional[str]:
        large_files = []
        try:
            for root, _, files in os.walk(self.base):
                for f in files:
                    if f.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c")):
                        fpath = Path(root) / f
                        size = fpath.stat().st_size
                        if size > 500 * 1024:
                            large_files.append((f, size))
        except:
            pass
        if len(large_files) >= 2:
            files_str = ", ".join([f[0] for f in large_files[:3]])
            return f"Found large files suggesting bloat: {files_str}. Switching to Optimizer."
        try:
            perf_patterns = [
                r"for\s+\w+\s+in\s+range\s*\(\s*len\s*\(",
                r"time\.sleep\s*\(\s*0\.0",
                r"SELECT\s+\*",
                r"\.append\s*\(\s*\)\s*in\s+loop",
            ]
            for pattern in perf_patterns:
                result = self.tools.search(pattern, ".")
                if not result.startswith("[INFO]"):
                    return f"Found performance anti-patterns. Switching to Optimizer."
        except:
            pass
        return None

    def _check_reviewer_signals(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.base,
                capture_output=True,
                text=True,
                shell=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                changed = len(result.stdout.strip().split("\n"))
                if changed >= 3:
                    return f"Git shows {changed} changed files. Ready for review. Switching to Reviewer."
        except:
            pass
        try:
            py_files = list(self.base.rglob("*.py"))
            if len(py_files) > 10:
                test_dirs = [d for d in self.base.rglob("*test*") if d.is_dir()]
                if test_dirs:
                    return f"Project has {len(py_files)} Python files and test suite. Switching to Reviewer."
        except:
            pass
        return None

    def _check_coder_signals(self) -> Optional[str]:
        todo_count = 0
        try:
            result = self.tools.search(r"TODO|FIXME|pass\s*$|NotImplementedError|raise.*implement", ".")
            if not result.startswith("[INFO]"):
                todo_count = result.count(":")
        except:
            pass
        if todo_count >= 3:
            return f"Found {todo_count} TODOs/incomplete stubs. Switching to Coder."
        try:
            config_files = list(self.base.glob("*.json")) + list(self.base.glob("*.yaml")) + list(self.base.glob("*.yml")) + list(self.base.glob("*.toml"))
            py_files = list(self.base.rglob("*.py"))
            if len(config_files) >= 1 and len(py_files) <= 3:
                return f"Project has config but minimal code ({len(py_files)} files). Switching to Coder."
        except:
            pass
        return None

    def _check_architect_signals(self) -> Optional[str]:
        try:
            items = list(self.base.iterdir())
            file_count = len([i for i in items if i.is_file()])
            dir_count = len([i for i in items if i.is_dir()])
            if file_count == 0 and dir_count == 0:
                return "Project directory is empty. Starting from scratch. Switching to Architect."
            if file_count <= 2 and dir_count <= 1:
                return f"Project is nearly empty ({file_count} files, {dir_count} dirs). Switching to Architect."
            md_files = list(self.base.glob("*.md"))
            py_files = list(self.base.rglob("*.py"))
            if len(md_files) >= 1 and len(py_files) == 0:
                return "Design docs exist but no code yet. Switching to Architect."
        except:
            pass
        return None


# ============================================
# FILE SYSTEM TOOLS
# ============================================

class FileTools:
    def __init__(self, base_path: str):
        self.base = Path(base_path).resolve()

    def _safe_path(self, path: str) -> Path:
        target = (self.base / path).resolve()
        if not str(target).startswith(str(self.base)):
            raise ValueError(f"Path '{path}' escapes project directory")
        return target

    def read(self, path: str) -> str:
        try:
            target = self._safe_path(path)
            if not target.exists():
                return f"[ERROR] File not found: {path}"
            if target.is_dir():
                return f"[ERROR] '{path}' is a directory, not a file"
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return f"=== {path} ===\n{content}\n=== END ==="
        except Exception as e:
            return f"[ERROR] {e}"

    def list_dir(self, path: str = ".") -> str:
        try:
            target = self._safe_path(path)
            if not target.exists():
                return f"[ERROR] Directory not found: {path}"
            if not target.is_dir():
                return f"[ERROR] '{path}' is not a directory"
            items = []
            for item in sorted(target.iterdir()):
                icon = "[DIR]" if item.is_dir() else "[FILE]"
                size = ""
                if item.is_file():
                    size_kb = item.stat().st_size / 1024
                    size = f" ({size_kb:.1f} KB)"
                items.append(f"  {icon} {item.name}{size}")
            return f"=== Directory: {path} ===\n" + "\n".join(items)
        except Exception as e:
            return f"[ERROR] {e}"

    def search(self, pattern: str, path: str = ".", ext: str = "") -> str:
        try:
            target = self._safe_path(path)
            matches = []
            regex = re.compile(pattern, re.IGNORECASE)
            for root, _, files in os.walk(target):
                for fname in files:
                    if ext and not fname.endswith(ext):
                        continue
                    fpath = Path(root) / fname
                    rel_path = fpath.relative_to(self.base)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            for i, line in enumerate(f, 1):
                                if regex.search(line):
                                    matches.append(f"  {rel_path}:{i}: {line.strip()}")
                                    if len(matches) >= 50:
                                        break
                    except:
                        pass
                    if len(matches) >= 50:
                        break
                if len(matches) >= 50:
                    break
            if not matches:
                return f"[INFO] No matches for '{pattern}'"
            return f"=== Search: '{pattern}' ===\n" + "\n".join(matches[:50])
        except Exception as e:
            return f"[ERROR] {e}"

    def tree(self, path: str = ".", max_depth: int = 3) -> str:
        try:
            target = self._safe_path(path)
            lines = [f"=== Tree: {path} ==="]
            def _tree(current: Path, prefix: str = "", depth: int = 0):
                if depth > max_depth:
                    return
                try:
                    items = sorted(current.iterdir(), key=lambda x: (not x.is_dir(), x.name))
                    for i, item in enumerate(items):
                        is_last = i == len(items) - 1
                        branch = "+-- " if is_last else "+-- "
                        lines.append(f"{prefix}{branch}{item.name}")
                        if item.is_dir():
                            extension = "    " if is_last else "|   "
                            _tree(item, prefix + extension, depth + 1)
                except PermissionError:
                    pass
            _tree(target)
            return "\n".join(lines)
        except Exception as e:
            return f"[ERROR] {e}"

    def write(self, path: str, content: str) -> str:
        try:
            target = self._safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[OK] Written: {path} ({len(content)} chars)"
        except Exception as e:
            return f"[ERROR] {e}"

    def git_status(self) -> str:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                cwd=self.base,
                capture_output=True,
                text=True,
                shell=True,
            )
            if result.returncode != 0:
                return "[ERROR] Not a git repository or git error"
            if not result.stdout.strip():
                return "[INFO] Working tree clean"
            return "=== Git Status ===\n" + result.stdout
        except Exception as e:
            return f"[ERROR] {e}"

    def git_diff(self) -> str:
        try:
            result = subprocess.run(
                ["git", "diff"],
                cwd=self.base,
                capture_output=True,
                text=True,
                shell=True,
            )
            if result.returncode != 0:
                return "[ERROR] Git diff failed"
            if not result.stdout.strip():
                return "[INFO] No changes to diff"
            return "=== Git Diff ===\n" + result.stdout[:5000]
        except Exception as e:
            return f"[ERROR] {e}"


# ============================================
# NIM CLIENT
# ============================================

class NIMClient:
    def __init__(self, profile: Dict, project_dir: str, restore_session: bool = True):
        self.profile = profile
        self.active_model_key = profile["default_model"]
        self.messages: List[Dict] = []
        self.goal: Optional[str] = None
        self.persona = PERSONAS["general"]
        self.tools = FileTools(project_dir)
        self.stage_detector = StageDetector(self.tools)
        self.auto_stage = True
        self.last_detected_stage = None
        self.project_dir = project_dir
        self._restored = False

        if HAS_OPENAI:
            self.client = OpenAI(
                base_url=NIM_BASE,
                api_key=profile["key"],
                timeout=120,
            )
        else:
            self.client = None

        # Try to restore previous session
        if restore_session:
            self._try_restore_session()
        else:
            self._auto_detect_stage()
            self._rebuild_system()

    def _try_restore_session(self):
        session = load_session(self.profile["name"], self.project_dir)
        if session:
            print(f"\n[Session Found] Previous session for {self.profile['name']} in this project.")
            print(f"  Last active: {session.get('timestamp', 'unknown')}")
            print(f"  Messages: {len(session.get('messages', []))}")
            print(f"  Goal: {session.get('goal', 'None')}")
            print(f"  Persona: {session.get('persona', 'General')}")
            resp = input("  Resume previous session? [Y/n]: ").strip().lower()
            if resp in ("", "y", "yes"):
                self.messages = session.get("messages", [])
                self.goal = session.get("goal")
                self.active_model_key = session.get("model_key", self.profile["default_model"])
                self.auto_stage = session.get("auto_stage", True)
                persona_key = session.get("persona_key", "general")
                if persona_key in PERSONAS:
                    self.persona = PERSONAS[persona_key]
                self.last_detected_stage = session.get("last_detected_stage")
                self._restored = True
                print(f"  [Restored] Session resumed.\n")
                self._rebuild_system()
                return
            else:
                delete_session(self.profile["name"], self.project_dir)
                print("  [New Session] Starting fresh.\n")
        self._auto_detect_stage()
        self._rebuild_system()

    def get_active_model(self) -> Dict:
        return self.profile["models"][self.active_model_key]

    def switch_model(self, model_key: str) -> bool:
        if model_key in self.profile["models"]:
            self.active_model_key = model_key
            self._rebuild_system()
            return True
        return False

    def _auto_detect_stage(self):
        if not self.auto_stage:
            return
        persona_key, reasoning = self.stage_detector.detect()
        if persona_key != self.last_detected_stage:
            self.last_detected_stage = persona_key
            self.persona = PERSONAS[persona_key]
            print(f"\n[Auto-Stage] {reasoning}")
            print(f"[Persona] {self.persona['name']} - {self.persona['description']}\n")

    def _rebuild_system(self):
        parts = [self.persona["prompt"]]
        if self.goal:
            parts.append(f"\nCURRENT GOAL: {self.goal}")
        parts.append(f"\nYou are working in project: {self.project_dir}")
        parts.append("\nYou have access to the file system. Use tools when needed.")
        self.messages = [m for m in self.messages if m.get("role") != "system"]
        self.messages.insert(0, {"role": "system", "content": "\n".join(parts)})

    def set_goal(self, goal: str):
        self.goal = goal
        self._rebuild_system()

    def set_persona(self, persona_key: str):
        if persona_key in PERSONAS:
            self.persona = PERSONAS[persona_key]
            self.auto_stage = False
            self._rebuild_system()
            return True
        return False

    def enable_auto_stage(self):
        self.auto_stage = True
        self._auto_detect_stage()
        self._rebuild_system()

    def chat(self, user_message: str, thinking: bool = True) -> tuple:
        if self.auto_stage:
            old_persona = self.persona["name"]
            self._auto_detect_stage()
            if self.persona["name"] != old_persona:
                self._rebuild_system()
        self.messages.append({"role": "user", "content": user_message})
        model = self.get_active_model()
        payload = {
            "model": model["id"],
            "messages": self.messages,
            "max_tokens": 4096,
            "temperature": 0.2,
        }
        if thinking and "glm" in model["id"]:
            if HAS_OPENAI:
                payload["extra_body"] = {
                    "chat_template_kwargs": {
                        "enable_thinking": True,
                        "clear_thinking": False,
                    }
                }
            else:
                payload["chat_template_kwargs"] = {
                    "enable_thinking": True,
                    "clear_thinking": False,
                }
        try:
            if HAS_OPENAI:
                response = self.client.chat.completions.create(**payload)
                content = response.choices[0].message.content
                reasoning = getattr(response.choices[0].message, "reasoning_content", None)
            else:
                content, reasoning = self._http_chat(payload)
            self.messages.append({"role": "assistant", "content": content})
            return content, reasoning
        except Exception as e:
            return f"[ERROR] {e}", None

    def _http_chat(self, payload: Dict) -> tuple:
        req = urllib.request.Request(
            f"{NIM_BASE}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.profile['key']}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return content, None

    def clear(self):
        self.messages = []
        self._rebuild_system()

    def save_state(self):
        """Save current session state for resumption."""
        # Find persona key from current persona
        persona_key = "general"
        for k, v in PERSONAS.items():
            if v["name"] == self.persona["name"]:
                persona_key = k
                break
        data = {
            "profile": self.profile["name"],
            "model_key": self.active_model_key,
            "persona_key": persona_key,
            "persona": self.persona["name"],
            "goal": self.goal,
            "auto_stage": self.auto_stage,
            "last_detected_stage": self.last_detected_stage,
            "timestamp": datetime.now().isoformat(),
            "messages": self.messages,
        }
        save_session(self.profile["name"], self.project_dir, data)

    def save_history(self):
        data = {
            "profile": self.profile["name"],
            "model": self.get_active_model()["name"],
            "persona": self.persona["name"],
            "goal": self.goal,
            "timestamp": datetime.now().isoformat(),
            "messages": self.messages,
        }
        histories = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                histories = json.load(f)
        histories.append(data)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(histories[-10:], f, indent=2, ensure_ascii=False)


# ============================================
# UI
# ============================================

def print_banner():
    print("=" * 60)
    print("  NIM Agent - Interactive Terminal Agent")
    print("  Auto Stage Detection | Goals | Personas | File Tools | Dual Models")
    print("  Context Persistence | API Key Persistence")
    print("=" * 60)
    print()

def print_profiles(profiles: Dict):
    print("Profiles (each has 2 models):")
    for key, p in profiles.items():
        glm = p["models"]["glm"]["name"]
        nem = p["models"]["nemotron"]["name"]
        has_key = "[KEY SET]" if p.get("key") else "[NO KEY]"
        print(f"  [{key}] {p['name']} {has_key}")
        print(f"      glm:      {glm}")
        print(f"      nemotron: {nem}")
    print()

def print_personas():
    print("Personas (auto-detected by default):")
    for key, p in PERSONAS.items():
        print(f"  /agent {key:<12} - {p['description']} [{p['stage']}]")
    print()

def print_help():
    print("""
GOALS & PERSONAS:
  /goal <text>        Set current project goal
  /agent <name>       Manual persona override (disables auto-detect)
  /agent auto         Re-enable automatic stage detection
  /goal               Show current goal
  /persona            Show current persona

MODELS (per project):
  /model              Show current active model
  /model glm          Switch to GLM-5.2
  /model nemotron     Switch to Nemotron Ultra

FILE SYSTEM TOOLS:
  /read <path>        Read file content
  /ls [path]          List directory contents
  /tree [path]        Show directory tree
  /search <pattern>   Search text in files (regex)
  /search <p> -e .py  Search only .py files
  /write <path>       Interactive file writer (multi-line)
  /git status         Show git status
  /git diff           Show git diff

SESSION & CONFIG:
  /switch <1-4>       Switch to another project profile
  /key <1-4> <key>    Update API key for a profile
  /keys               Show which profiles have keys set
  /save               Save conversation to history file
  /clear              Clear conversation history (keeps session)
  /forget             Delete saved session for this project
  /history            Show conversation history
  /help               Show this help
  /exit               Exit (auto-saves session)

Just type your message to chat with the AI.
""")

def prompt_multiline() -> str:
    print("Enter content (empty line to finish):")
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)

def main():
    print_banner()
    profiles = get_profiles()
    print_profiles(profiles)

    choice = input("Select profile [1-4] (default: 1): ").strip() or "1"
    if choice not in profiles:
        print("Invalid choice. Using profile 1.")
        choice = "1"
        if choice not in profiles:
            print("[ERROR] Profile 1 has no API key. Use /key to set one.")
            sys.exit(1)

    profile = profiles[choice]
    project_dir = os.getcwd()
    agent = NIMClient(profile, project_dir, restore_session=True)
    model = agent.get_active_model()

    print(f"[Active] {profile['name']} - {model['name']} ({model['id']})")
    print(f"Project: {project_dir}")
    print("Models: /model glm | /model nemotron")
    print_personas()
    print("Type /help for commands. Type your message to chat.\n")

    try:
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                break

            if not user_input:
                continue

            if user_input.startswith("/exit"):
                print("Goodbye!")
                break

            elif user_input.startswith("/help"):
                print_help()

            elif user_input.startswith("/switch "):
                new_choice = user_input.split()[1] if len(user_input.split()) > 1 else ""
                if new_choice in profiles:
                    agent.save_state()
                    profile = profiles[new_choice]
                    agent = NIMClient(profile, project_dir, restore_session=True)
                    model = agent.get_active_model()
                    print(f"\n[Switched] {profile['name']} - {model['name']}")
                    print(f"Project: {project_dir}\n")
                else:
                    print("Invalid profile. Use /switch 1, /switch 2, etc.")

            elif user_input.startswith("/key "):
                parts = user_input.split(None, 2)
                if len(parts) >= 3:
                    pid = parts[1].strip()
                    new_key = parts[2].strip()
                    if update_key(pid, new_key):
                        profiles = get_profiles()
                        print(f"[Updated] API key for profile [{pid}] saved.")
                    else:
                        print("Invalid profile ID.")
                else:
                    print("Usage: /key <1-4> <api_key>")

            elif user_input == "/keys":
                for k, p in profiles.items():
                    status = "SET" if p.get("key") else "NOT SET"
                    print(f"  [{k}] {p['name']}: {status}")

            elif user_input.startswith("/goal "):
                goal = user_input[6:].strip()
                agent.set_goal(goal)
                print(f"[Goal set] {goal}")
            elif user_input == "/goal":
                if agent.goal:
                    print(f"[Current Goal] {agent.goal}")
                else:
                    print("[No goal set] Use /goal <description>")

            elif user_input.startswith("/agent "):
                persona_key = user_input[7:].strip()
                if persona_key == "auto":
                    agent.enable_auto_stage()
                    print(f"[Auto-Stage ON] Current: {agent.persona['name']} - {agent.persona['description']}")
                elif agent.set_persona(persona_key):
                    print(f"[Manual Persona] {agent.persona['name']} - {agent.persona['description']}")
                    print("[Auto-Stage OFF] Use '/agent auto' to re-enable.")
                else:
                    print("Invalid persona. Options: coder, architect, reviewer, debugger, optimizer, general")
                    print("Use '/agent auto' to re-enable automatic detection.")
            elif user_input == "/persona":
                mode = "[AUTO]" if agent.auto_stage else "[MANUAL]"
                print(f"{mode} {agent.persona['name']} - {agent.persona['description']}")

            elif user_input.startswith("/model "):
                model_key = user_input[7:].strip()
                if agent.switch_model(model_key):
                    model = agent.get_active_model()
                    print(f"[Model switched] {model['name']} ({model['id']})")
                else:
                    print("Invalid model. Use: /model glm or /model nemotron")
            elif user_input == "/model":
                model = agent.get_active_model()
                print(f"[Current Model] {model['name']} ({model['id']})")
                print("Available:")
                for k, m in agent.profile["models"].items():
                    marker = " <-- active" if k == agent.active_model_key else ""
                    print(f"  /model {k:<10} - {m['name']}{marker}")

            elif user_input.startswith("/read "):
                path = user_input[6:].strip().strip('"')
                print(agent.tools.read(path))

            elif user_input.startswith("/ls"):
                path = user_input[3:].strip().strip('"') or "."
                print(agent.tools.list_dir(path))

            elif user_input.startswith("/tree"):
                path = user_input[5:].strip().strip('"') or "."
                print(agent.tools.tree(path))

            elif user_input.startswith("/search "):
                parts = user_input[8:].strip().split(" -e ")
                pattern = parts[0].strip('"')
                ext = parts[1] if len(parts) > 1 else ""
                print(agent.tools.search(pattern, ext=ext))

            elif user_input.startswith("/write "):
                path = user_input[7:].strip().strip('"')
                content = prompt_multiline()
                if content:
                    print(f"\nPreview ({len(content)} chars):")
                    print(content[:200] + "..." if len(content) > 200 else content)
                    confirm = input("\nWrite to file? [y/N]: ").strip().lower()
                    if confirm == "y":
                        print(agent.tools.write(path, content))
                    else:
                        print("[Cancelled]")
                else:
                    print("[No content provided]")

            elif user_input.startswith("/git "):
                subcmd = user_input[5:].strip()
                if subcmd == "status":
                    print(agent.tools.git_status())
                elif subcmd == "diff":
                    print(agent.tools.git_diff())
                else:
                    print("Unknown git command. Use: /git status, /git diff")

            elif user_input.startswith("/clear"):
                agent.clear()
                print("[History cleared]")

            elif user_input.startswith("/forget"):
                delete_session(agent.profile["name"], agent.project_dir)
                print("[Session deleted] Next launch will start fresh.")

            elif user_input.startswith("/history"):
                for i, msg in enumerate(agent.messages):
                    if msg.get("role") == "system":
                        continue
                    role = msg["role"]
                    content = msg["content"][:80].replace("\n", " ")
                    print(f"  [{i}] {role}: {content}...")

            elif user_input.startswith("/save"):
                agent.save_history()
                print(f"[Saved to {HISTORY_FILE}]")

            else:
                print("[Sending...]")
                response, reasoning = agent.chat(user_input)
                print("\n" + "=" * 60)
                if reasoning:
                    print("[Thinking]")
                    print(reasoning)
                    print("-" * 60)
                print(response)
                print("=" * 60 + "\n")
    finally:
        # Always save session on exit
        agent.save_state()
        print(f"[Session saved] Resumable on next launch.")

if __name__ == "__main__":
    main()
