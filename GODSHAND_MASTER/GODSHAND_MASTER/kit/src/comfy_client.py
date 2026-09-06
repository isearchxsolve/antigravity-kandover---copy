#!/usr/bin/env python3
"""Minimal headless ComfyUI API client (no tunnel needed).
Talks to a local ComfyUI server started by the notebook / run.sh.
Queues a prompt graph, waits for completion, fetches outputs. Retries + resumable.

Graphs are ComfyUI API-format node maps (node_id -> {class_type, inputs}).
Editor-only top-level keys that start with '_' (e.g. '_bind', '_meta') are
STRIPPED before POST. '_bind' routes the pipeline's semantic override keys
(positive, negative, width, height, seconds, steps, cfg, seed, ref_dirs,
identity_images, prompt) onto concrete [node_id, input_name] targets, so the
render/forge/de-age stages can keep passing the same override dict they always did.
"""
import glob, json, os, subprocess, time, urllib.request, urllib.parse, uuid

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


class ComfyClient:
    def __init__(self, host=None, port=None, fps=24):
        host = host or os.environ.get("COMFY_HOST", "127.0.0.1")
        port = int(port or os.environ.get("COMFY_PORT", "8188"))
        self.base = f"http://{host}:{port}"
        self.cid = str(uuid.uuid4())
        self.fps = fps
        # COMFY_MOCK=1 -> no server; synthesize clips with ffmpeg so the whole
        # orchestration can be dry-run offline for FREE (see src/selftest.py).
        self.mock = os.environ.get("COMFY_MOCK") == "1"
        
        # AGGRESSIVE CUDA MEMORY SETTINGS FOR ZERO OOM ON T4
        # These must be set BEFORE any torch import
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
            'expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6')
        os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
        os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 
            'expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.6')

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=120))

    def _get(self, path):
        return json.load(urllib.request.urlopen(self.base + path, timeout=120))

    @staticmethod
    def _first_image(d):
        if not d or not os.path.isdir(d):
            return None
        for f in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
            # Skip staging/junk folders like refs/<name>/_inbox (underscore-prefixed dirs)
            if any(p.startswith("_") for p in os.path.relpath(f, d).split(os.sep)[:-1]):
                continue
            if f.lower().endswith(IMG_EXT):
                return f
        return None

    def apply_overrides(self, graph, overrides):
        """Route semantic overrides onto concrete nodes using graph['_bind'].""" 
        bind = graph.get("_bind", {})
        for key, val in (overrides or {}).items():
            target = bind.get(key)
            if not target:
                continue
            node_id, input_name = target
            node = graph.get(node_id, {}).get("inputs")
            if node is None:
                continue
            if key == "seconds":
                val = int(round(float(val) * self.fps)) + 1  # seconds -> frame count
            elif key == "ref_dirs" and isinstance(val, dict):
                img = None
                for d in val.values():
                    img = self._first_image(d)
                    if img:
                        break
                if not img:
                    continue
                val = img
            elif key == "identity_images" and isinstance(val, (list, tuple)):
                if not val:
                    continue
                val = val[0]  # core LoadImage takes one file; add nodes for more anchors
            node[input_name] = val
        return graph

    @staticmethod
    def _clean(graph):
        """Strip editor-only keys ('_bind', '_meta', ...) before sending to ComfyUI."""
        return {k: v for k, v in graph.items() if not k.startswith("_")}

    def run(self, graph, overrides=None, tries=3):
        if getattr(self, "mock", False):
            return {"_mock": {"videos": []}}
        graph = json.loads(json.dumps(graph))  # deep copy; never mutate the loaded file
        if overrides:
            graph = self.apply_overrides(graph, overrides)
        prompt = self._clean(graph)
        last = None
        for attempt in range(tries):
            try:
                r = self._post("/prompt", {"prompt": prompt, "client_id": self.cid})
                pid = r["prompt_id"]
                while True:
                    hist = self._get(f"/history/{pid}")
                    if pid in hist:
                        return hist[pid]["outputs"]
                    time.sleep(2)
            except Exception as e:
                last = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"ComfyUI run failed after {tries} tries: {last}")

    def save_image(self, outputs, dest):
        """Fetch the first produced media (image OR video) and write it to dest."""
        if getattr(self, "mock", False):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if dest.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # Still image: synthesize a single PNG frame
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                                "-i", "testsrc2=size=512x288:rate=1:duration=1",
                                "-frames:v", "1", dest], check=True)
            else:
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                                "-i", "testsrc2=size=512x288:rate=24:duration=2",
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", dest], check=True)
            return dest
        for node_out in outputs.values():
            for kind in ("images", "gifs", "videos"):
                for media in node_out.get(kind, []):
                    q = urllib.parse.urlencode({"filename": media["filename"],
                                                "subfolder": media.get("subfolder", ""),
                                                "type": media.get("type", "output")})
                    raw = urllib.request.urlopen(self.base + "/view?" + q, timeout=120).read()
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    open(dest, "wb").write(raw)
                    return dest
        raise RuntimeError("No image/video in ComfyUI outputs")
