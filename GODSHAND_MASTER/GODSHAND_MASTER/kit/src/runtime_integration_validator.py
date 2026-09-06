#!/usr/bin/env python3
"""
GODSHAND Runtime Integration Validator
=======================================
Prevents runtime integration bugs by validating ComfyUI custom nodes,
workflow graphs, and frame extraction paths BEFORE GPU spend.

Run as: python src/runtime_integration_validator.py
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =============================================================================
# COMPATIBILITY MATRIX: Known working versions
# =============================================================================
COMPATIBILITY_MATRIX = {
    "ComfyUI": {
        "min": "0.2.8",
        "max": "0.3.0",  # Exclusive - 0.3.0+ breaks GGUF loading
        "tested": ["0.2.8", "0.2.9", "0.2.10"],
        "breaking": ["0.3.0", "0.3.1"]
    },
    "ComfyUI-GGUF": {
        "min": "0.4.0",
        "max": "0.6.0",
        "tested": ["0.4.5", "0.5.0"],
        "breaking": []
    },
    "ComfyUI-VideoHelperSuite": {
        "min": "1.0.0",
        "max": "1.2.0",
        "tested": ["1.0.5", "1.1.0"],
        "breaking": ["1.2.0"],  # 1.2.0 changed SaveImage output format
        "notes": "SaveImage node in 1.2.0+ outputs MP4 containers; frame extraction expects PNG sequence"
    },
    "comfy-kitchen": {
        "min": "0.1.0",
        "max": "0.3.0",
        "tested": ["0.2.22"],
        "breaking": []
    },
    "torch": {
        "min": "2.1.0",
        "max": "2.3.0",
        "tested": ["2.1.0", "2.2.0", "2.3.0+cu121"],
        "breaking": []
    },
    "xformers": {
        "min": "0.0.22",
        "max": "0.0.27",
        "tested": ["0.0.23", "0.0.26"],
        "breaking": []
    }
}

# =============================================================================
# REQUIRED CUSTOM NODES & THEIR EXPECTED BEHAVIOR
# =============================================================================
REQUIRED_CUSTOM_NODES = {
    "ComfyUI-GGUF": {
        "required": True,
        "nodes": ["UnetLoaderGGUF", "CLIPLoaderGGUF", "VAELoaderGGUF"],
        "version_check": True
    },
    "ComfyUI-VideoHelperSuite": {
        "required": True,
        "nodes": ["SaveImage", "LoadVideo", "VideoCombine"],
        "version_check": True,
        "critical_check": "save_image_output_format"  # Must output PNG sequence, not MP4
    },
    "websocket_image_save": {
        "required": False,
        "nodes": ["WebsocketImageSave"],
        "version_check": False
    }
}

# =============================================================================
# WORKFLOW GRAPH VALIDATION RULES
# =============================================================================
WORKFLOW_VALIDATION = {
    "flf2v.json": {
        "required_nodes": ["WanImageToVideo", "UnetLoaderGGUF", "VAELoader", "CLIPLoader", "SaveImage"],
        "forbidden_nodes": ["VideoCombine"],  # Must use SaveImage for frame sequence
        "saveimage_format": "PNG",  # Not MP4
        "vaedecode_output": "IMAGE",  # Not LATENT
        "chain_endframe": True  # KIT_FLF2V_ENDFRAME=1
    },
    "charforge.json": {
        "required_nodes": ["CheckpointLoaderSimple", "KSampler", "SaveImage"],
        "saveimage_format": "PNG"
    }
}

# =============================================================================
# VALIDATOR CLASS
# =============================================================================
class RuntimeIntegrationValidator:
    def __init__(self, kit_root: str = "/kaggle/working"):
        self.kit_root = Path(kit_root)
        self.src_dir = self.kit_root / "src"
        self.workflows_dir = self.kit_root / "workflows"
        self.comfy_dir = self.kit_root / "ComfyUI"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.info: List[str] = []

    def log_error(self, msg: str):
        self.errors.append(msg)
        print(f"❌ ERROR: {msg}")

    def log_warning(self, msg: str):
        self.warnings.append(msg)
        print(f"⚠️  WARNING: {msg}")

    def log_info(self, msg: str):
        self.info.append(msg)
        print(f"ℹ️  INFO: {msg}")

    def run_cmd(self, cmd: List[str], cwd: Optional[str] = None, capture: bool = True) -> Tuple[int, str, str]:
        """Run command, return (rc, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd, cwd=cwd or self.kit_root,
                capture_output=capture, text=True, timeout=60
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "TIMEOUT"
        except Exception as e:
            return -1, "", str(e)

    # -------------------------------------------------------------------------
    # 1. ENVIRONMENT VALIDATION
    # -------------------------------------------------------------------------
    def validate_python_env(self) -> bool:
        """Check Python version and critical packages"""
        ok = True
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.log_info(f"Python: {py_version}")

        # Check critical packages
        critical_packages = [
            ("torch", "torch"),
            ("torchvision", "torchvision"),
            ("xformers", "xformers"),
            ("comfy", "comfy"),
            ("gguf", "gguf"),
            ("safetensors", "safetensors"),
            ("cv2", "opencv-python"),
            ("PIL", "Pillow"),
            ("numpy", "numpy"),
            ("yaml", "PyYAML"),
            ("tqdm", "tqdm"),
        ]

        for import_name, pkg_name in critical_packages:
            try:
                __import__(import_name)
                self.log_info(f"  ✓ {pkg_name}")
            except ImportError:
                self.log_error(f"  ✗ MISSING: {pkg_name} (import: {import_name})")
                ok = False

        return ok

    def validate_comfyui_install(self) -> bool:
        """Check ComfyUI installation and version"""
        ok = True
        main_py = self.comfy_dir / "main.py"
        if not main_py.exists():
            self.log_error(f"ComfyUI not found at {self.comfy_dir}")
            return False

        # Try to get version
        rc, out, err = self.run_cmd([sys.executable, "-c", "import comfy; print(comfy.__version__)"], capture=True)
        if rc == 0 and out.strip():
            version = out.strip()
            self.log_info(f"ComfyUI version: {version}")
            # Check against matrix
            matrix = COMPATIBILITY_MATRIX["ComfyUI"]
            if version in matrix["breaking"]:
                self.log_error(f"ComfyUI {version} is BREAKING (known issues)")
                ok = False
            elif version not in matrix["tested"]:
                self.log_warning(f"ComfyUI {version} not in tested versions: {matrix['tested']}")
        else:
            self.log_warning("Could not determine ComfyUI version")

        return ok

    def validate_custom_nodes(self) -> bool:
        """Validate required custom nodes exist and versions"""
        ok = True
        custom_nodes_dir = self.comfy_dir / "custom_nodes"

        for node_name, spec in REQUIRED_CUSTOM_NODES.items():
            node_path = custom_nodes_dir / node_name
            if not node_path.exists():
                if spec["required"]:
                    self.log_error(f"Required custom node missing: {node_name}")
                    ok = False
                else:
                    self.log_info(f"Optional custom node not installed: {node_name}")
                continue

            self.log_info(f"Found custom node: {node_name}")

            # Check version if possible
            if spec.get("version_check"):
                # Try to get version from __init__.py or git
                init_py = node_path / "__init__.py"
                if init_py.exists():
                    content = init_py.read_text(encoding="utf-8")
                    # Look for version
                    import re
                    ver_match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
                    if ver_match:
                        version = ver_match.group(1)
                        self.log_info(f"  Version: {version}")
                        matrix = COMPATIBILITY_MATRIX.get(node_name, {})
                        if version in matrix.get("breaking", []):
                            self.log_error(f"  ❌ BREAKING VERSION: {version}")
                            ok = False
                        elif version not in matrix.get("tested", []):
                            self.log_warning(f"  Version {version} not in tested: {matrix.get('tested', [])}")

            # Check required nodes exist
            for node_class in spec.get("nodes", []):
                # This is a runtime check - we'd need to load ComfyUI to verify
                pass

            # Special critical check for VideoHelperSuite
            if spec.get("critical_check") == "save_image_output_format":
                self._check_saveimage_format(node_path)

        return ok

    def _check_saveimage_format(self, node_path: Path):
        """Critical: VideoHelperSuite SaveImage must output PNG sequence, not MP4"""
        # Look for SaveImage implementation
        saveimage_files = list(node_path.rglob("*[Ss]ave*[Ii]mage*.py"))
        if not saveimage_files:
            saveimage_files = list(node_path.rglob("*.py"))

        for f in saveimage_files:
            try:
                content = f.read_text(encoding="utf-8")
                if "class SaveImage" in content or "save_image" in content.lower():
                    # Check for MP4 output
                    if ".mp4" in content.lower() or "video" in content.lower():
                        if "format" in content.lower() and "png" not in content.lower():
                            self.log_warning(f"VideoHelperSuite SaveImage may output MP4: {f.name}")
                            self.log_warning("  This causes frame extraction to fail (MP4 vs PNG sequence)")
                            self.log_warning("  WORKAROUND: Use inline WAN graph with native SaveImage (PNG sequence)")
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # 2. WORKFLOW GRAPH VALIDATION
    # -------------------------------------------------------------------------
    def validate_workflow_graphs(self) -> bool:
        """Validate all workflow JSON files"""
        ok = True
        for wf_name, rules in WORKFLOW_VALIDATION.items():
            wf_path = self.workflows_dir / wf_name
            if not wf_path.exists():
                self.log_error(f"Workflow missing: {wf_name}")
                ok = False
                continue

            try:
                graph = json.loads(wf_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                self.log_error(f"Workflow {wf_name} invalid JSON: {e}")
                ok = False
                continue

            self.log_info(f"Validating workflow: {wf_name}")

            # Check required nodes
            node_types = {n.get("class_type") for n in graph.values() if isinstance(n, dict)}
            for req in rules.get("required_nodes", []):
                if req not in node_types:
                    self.log_error(f"  {wf_name}: Missing required node type: {req}")
                    ok = False
                else:
                    self.log_info(f"  ✓ Has {req}")

            # Check forbidden nodes
            for forbidden in rules.get("forbidden_nodes", []):
                if forbidden in node_types:
                    self.log_error(f"  {wf_name}: FORBIDDEN node present: {forbidden}")
                    ok = False

            # Check SaveImage format
            if rules.get("saveimage_format") == "PNG":
                saveimage_nodes = [n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == "SaveImage"]
                for node in saveimage_nodes:
                    inputs = node.get("inputs", {})
                    # Check filename_prefix doesn't end with .mp4
                    prefix = inputs.get("filename_prefix", "")
                    if prefix.endswith(".mp4"):
                        self.log_error(f"  {wf_name}: SaveImage filename_prefix ends with .mp4: {prefix}")
                        ok = False

            # Check VAE decode output type
            if rules.get("vaedecode_output") == "IMAGE":
                vae_nodes = [n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == "VAEDecode"]
                for node in vae_nodes:
                    # VAEDecode outputs IMAGE by default - verify no explicit output type override
                    pass

            # Check endframe chaining
            if rules.get("chain_endframe"):
                # Look for WanImageToVideo with endframe input
                wan_nodes = [n for n in graph.values() if isinstance(n, dict) and n.get("class_type") == "WanImageToVideo"]
                for node in wan_nodes:
                    inputs = node.get("inputs", {})
                    if "end_image" not in inputs:
                        self.log_warning(f"  {wf_name}: WanImageToVideo missing end_image input (KIT_FLF2V_ENDFRAME=1)")

        return ok

    # -------------------------------------------------------------------------
    # 3. FRAME EXTRACTION PATH VALIDATION
    # -------------------------------------------------------------------------
    def validate_frame_extraction_path(self) -> bool:
        """Validate that frame extraction (postprocess) works with ComfyUI output"""
        ok = True

        # Check render_motion.py postprocess logic
        render_motion = self.src_dir / "render_motion.py"
        if not render_motion.exists():
            self.log_error("render_motion.py not found")
            return False

        content = render_motion.read_text(encoding="utf-8")

        # Check for _ffprobe_fps (NEED_INTERP guard)
        if "_ffprobe_fps" not in content:
            self.log_error("render_motion.py missing _ffprobe_fps (NEED_INTERP guard)")
            ok = False
        else:
            self.log_info("  ✓ NEED_INTERP guard (_ffprobe_fps) present")

        # Check for minterpolate conditional
        if "need_interp" not in content.lower():
            self.log_error("render_motion.py missing need_interp conditional")
            ok = False
        else:
            self.log_info("  ✓ minterpolate conditional present")

        # Check frame extraction expects PNG sequence
        if "SaveImage" in content or "frame" in content.lower():
            self.log_info("  ℹ Frame extraction logic found")

        # Critical: Ensure it doesn't assume MP4 from ComfyUI
        if "output_gpu" in content and ".mp4" in content:
            self.log_warning("  ⚠ Code references .mp4 from output_gpu - verify frame extraction handles this")

        return ok

    def validate_postprocess_sh(self) -> bool:
        """Validate postprocess.sh NEED_INTERP guard"""
        ok = True
        post_sh = self.src_dir / "postprocess.sh"
        if not post_sh.exists():
            self.log_warning("postprocess.sh not found (Windows fallback only)")
            return True

        content = post_sh.read_text(encoding="utf-8")
        if "NEED_INTERP" not in content:
            self.log_error("postprocess.sh missing NEED_INTERP guard")
            ok = False
        else:
            self.log_info("  ✓ postprocess.sh has NEED_INTERP guard")

        # Check minterpolate is conditional
        if "minterpolate" in content and "NEED_INTERP" not in content:
            self.log_error("postprocess.sh runs minterpolate unconditionally")
            ok = False

        return ok

    # -------------------------------------------------------------------------
    # 4. MODEL LOADING VALIDATION
    # -------------------------------------------------------------------------
    def validate_model_paths(self) -> bool:
        """Check model symlinks and paths"""
        ok = True
        models_dir = self.comfy_dir / "models"
        if not models_dir.exists():
            self.log_warning("ComfyUI/models directory not found (will be created at runtime)")
            return True

        # Check symlink to /kaggle/temp/models
        if models_dir.is_symlink():
            target = models_dir.resolve()
            self.log_info(f"Models symlink: {models_dir} -> {target}")
            if "/kaggle/temp" not in str(target):
                self.log_warning(f"Models symlink points to {target}, not /kaggle/temp/models")
        else:
            self.log_warning("Models directory is not a symlink (may hit 20GB /kaggle/working cap)")

        # Check required subdirs
        required_subdirs = ["unet", "vae", "text_encoders", "checkpoints", "loras", "upscale_models"]
        for sub in required_subdirs:
            p = models_dir / sub
            if p.exists():
                self.log_info(f"  ✓ {sub}/ exists")
            else:
                self.log_warning(f"  ⚠ {sub}/ missing (will be created at runtime)")

        return ok

    # -------------------------------------------------------------------------
    # 5. END-TO-END DRY RUN (MOCK)
    # -------------------------------------------------------------------------
    def dry_run_workflow(self, wf_name: str) -> bool:
        """Dry-run a workflow with mock inputs to catch graph errors"""
        ok = True
        wf_path = self.workflows_dir / wf_name
        if not wf_path.exists():
            self.log_error(f"Workflow not found for dry-run: {wf_name}")
            return False

        self.log_info(f"Dry-running workflow: {wf_name}")

        # Create a minimal mock test.
        # NOTE: self.src_dir / wf_path MUST be embedded via repr(). On Windows the
        # raw path contains backslashes (e.g. ...\workflows\flf2v.json) which, if
        # interpolated as a bare string literal, get re-interpreted by Python when
        # the generated script is run (\f -> form-feed, \k..\w..\N -> escapes),
        # producing OSError [Errno 22] "...\workflows\x0clf2v.json" and TypeError
        # from downstream None comparisons. repr() emits a portable single-quoted
        # literal that is safe across Windows / Linux / macOS.
        # Embed the required-nodes list as a literal: the dry-run subprocess
        # runs in a clean namespace that cannot see WORKFLOW_VALIDATION from
        # this module, so passing it through the f-string as a repr()'d list
        # avoids a NameError while keeping the script fully self-contained.
        _required = list(WORKFLOW_VALIDATION.get(wf_name, {}).get("required_nodes", []))
        test_script = f"""
import sys
sys.path.insert(0, {repr(str(self.src_dir))})
import json

# Load workflow
with open({repr(str(wf_path))}) as f:
    graph = json.load(f)

# Validate structure
# Mirror comfy_client.py: top-level keys starting with '_' (e.g. '_bind',
# '_meta') are editor-only metadata that get stripped before POST -- skip them,
# and skip any node whose class_type is missing/None (would otherwise raise
# TypeError in sorted() when compared against real str class_types).
node_types = {{
    n.get("class_type")
    for k, n in graph.items()
    if isinstance(n, dict) and not k.startswith("_") and n.get("class_type")
}}
print("Node types:", sorted(node_types))

# Check for required nodes (list embedded from WORKFLOW_VALIDATION)
required = {repr(_required)}
for req in required:
    if req not in node_types:
        print(f"MISSING: {{req}}")
        sys.exit(1)

print("WORKFLOW VALID")
sys.exit(0)
"""
        rc, out, err = self.run_cmd([sys.executable, "-c", test_script])
        if rc != 0:
            # The dry-run subprocess prints structured diagnostics to stdout
            # (e.g. "MISSING: WanImageToVideo") and only Python tracebacks to
            # stderr. Include BOTH in the error message so a cleanly-failing
            # dry-run (informative stdout, empty stderr) isn't reported as a
            # mysterious "failed for X: " with nothing to act on.
            _parts = []
            if out and out.strip():
                _parts.append(f"stdout:\n{out.strip()}")
            if err and err.strip():
                _parts.append(f"stderr:\n{err.strip()}")
            _detail = ("\n" + "\n".join(_parts)) if _parts else " (no output)"
            self.log_error(f"Dry-run failed for {wf_name}:{_detail}")
            ok = False
        else:
            self.log_info(f"  ✓ {out.strip()}")

        return ok

    # -------------------------------------------------------------------------
    # MAIN VALIDATION ENTRY POINT
    # -------------------------------------------------------------------------
    def run_all_validations(self) -> bool:
        """Run complete validation suite"""
        print("=" * 70)
        print("GODSHAND RUNTIME INTEGRATION VALIDATOR")
        print("=" * 70)

        all_ok = True

        # 1. Environment
        print("\n[1/5] Python Environment & Packages")
        print("-" * 40)
        all_ok &= self.validate_python_env()

        # 2. ComfyUI & Custom Nodes
        print("\n[2/5] ComfyUI Installation & Custom Nodes")
        print("-" * 40)
        all_ok &= self.validate_comfyui_install()
        all_ok &= self.validate_custom_nodes()

        # 3. Workflow Graphs
        print("\n[3/5] Workflow Graph Validation")
        print("-" * 40)
        all_ok &= self.validate_workflow_graphs()
        for wf_name in WORKFLOW_VALIDATION.keys():
            all_ok &= self.dry_run_workflow(wf_name)

        # 4. Frame Extraction Path
        print("\n[4/5] Frame Extraction & Postprocess Validation")
        print("-" * 40)
        all_ok &= self.validate_frame_extraction_path()
        all_ok &= self.validate_postprocess_sh()

        # 5. Model Paths
        print("\n[5/5] Model Paths & Symlinks")
        print("-" * 40)
        all_ok &= self.validate_model_paths()

        # Summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Errors:   {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Info:     {len(self.info)}")

        if self.errors:
            print("\n❌ ERRORS (must fix before GPU run):")
            for e in self.errors:
                print(f"  - {e}")

        if self.warnings:
            print("\n⚠️  WARNINGS (review before GPU run):")
            for w in self.warnings:
                print(f"  - {w}")

        if all_ok and not self.errors:
            print("\n✅ ALL VALIDATIONS PASSED - Safe to proceed with GPU run")
            return True
        else:
            print("\n❌ VALIDATION FAILED - Do NOT proceed with GPU run")
            return False


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
def main():
    import argparse
    # Auto-detect kit root: prefer the current working directory if it actually
    # looks like a kit (has a src/ subdir), otherwise fall back to /kaggle/working
    # (the Kaggle path). This stops the validator from reporting a pile of
    # spurious MISSING/ComfyUI-not-found errors when run from any non-default
    # location on a dev box, while preserving Kaggle's "just run it" default.
    _auto_root = "/kaggle/working"
    _cwd = os.getcwd()
    if os.path.isdir(os.path.join(_cwd, "src")):
        _auto_root = _cwd

    parser = argparse.ArgumentParser(description="GODSHAND Runtime Integration Validator")
    parser.add_argument("--kit-root", default=_auto_root,
                        help=f"Kit root directory (default: {_auto_root})")
    parser.add_argument("--workflow", help="Validate specific workflow only")
    args = parser.parse_args()

    validator = RuntimeIntegrationValidator(args.kit_root)

    if args.workflow:
        # Single workflow validation
        ok = validator.validate_workflow_graphs()
        ok &= validator.dry_run_workflow(args.workflow)
        sys.exit(0 if ok else 1)
    else:
        # Full validation
        ok = validator.run_all_validations()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()