#!/usr/bin/env python3
"""Shared config loader. Loads config.yaml and merges the selected hardware
profile (config['active_profile']) onto the top-level keys, so render / stitch /
cost-meter all see one effective config. Keeps config.yaml itself pristine."""
import os

def load_cfg(root):
    import yaml
    cfg_name = os.environ.get("KIT_CONFIG", "config.yaml")  # Film II sets KIT_CONFIG=config_observer.yaml
    cfg = yaml.safe_load(open(os.path.join(root, cfg_name)))
    cfg["_config_file"] = cfg_name
    name = cfg.get("active_profile")
    override = os.environ.get("KIT_PROFILE")  # explicit hardware override, e.g. kaggle_2xt4
    if override:
        name = override
        cfg["active_profile"] = override
    profiles = cfg.get("profiles", {}) or {}
    prof = profiles.get(name) if name else None
    if prof:
        for k, v in prof.items():
            cfg[k] = v
        cfg["_active_profile"] = name
    else:
        cfg["_active_profile"] = "(base)"
    return cfg
