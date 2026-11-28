"""Named motor lookup in the actuator catalogue.

Setting e.g. ``L_motor: AK80-9 V3`` in parameters.yaml fills the class's peak
torque, no-load speed and gear ratio from the catalogue; explicit fields win.
Size and mass are derived by the trends in ``feasibility.py``, not copied.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Optional

_CATALOG: dict[str, list[dict]] = {}


# ── Catalog I/O ────────────────────────────────────────────────────────────────

def load_catalog() -> list[dict]:
    """Load and cache ``actuator_catalog.json`` (the rows the trends are fitted on).

    Returns [] if it is missing.
    """
    from .feasibility import CATALOG_JSON
    if "rows" not in _CATALOG:
        if not CATALOG_JSON.exists():
            warnings.warn(
                f"Motor catalog not found at {CATALOG_JSON}. "
                "Named motor lookup and validation disabled.",
                UserWarning, stacklevel=3,
            )
            _CATALOG["rows"] = []
        else:
            with CATALOG_JSON.open(encoding="utf-8") as f:
                _CATALOG["rows"] = json.load(f)
    return _CATALOG["rows"]


def _norm(name: str) -> str:
    """Normalize a motor name for case/dash/space-insensitive lookup."""
    return name.lower().replace(" ", "_").replace("-", "_")


def with_robot_catalog(params: dict, robot_dir: Path) -> dict:
    """Load robot-local supplied specs without extending the paper population."""
    path = Path(robot_dir) / 'motor_catalog.json'
    if not path.is_file():
        return params
    raw = json.loads(path.read_text(encoding='utf-8'))
    if raw.get('schema_version') != 1 or not isinstance(raw.get('motors'), list):
        raise ValueError(f'{path}: expected schema_version 1 and a motors list')
    local = {}
    for spec in raw['motors']:
        entry = supplied_entry(spec)
        if entry['name'] in local:
            raise ValueError(f'{path}: duplicate motor {entry["name"]}')
        local[entry['name']] = entry
    return {**params, '_motor_catalog': local}


def supplied_entry(spec: dict) -> dict:
    """Validate SI conversions and keep the original approximate digitization."""
    name = spec.get('name', '<unnamed>')
    if not isinstance(name, str) or not name.strip() or name == '<unnamed>':
        raise ValueError('supplied motor requires a nonempty name')
    for key in ('continuous_torque', 'peak_torque_10s', 'peak_torque_3s',
                'no_load_speed_rpm', 'gear_ratio', 'mass'):
        if not isinstance(spec.get(key), (int, float)) or not math.isfinite(spec[key]) or spec[key] <= 0:
            raise ValueError(f'{name}: {key} must be positive and finite')
    if not spec['continuous_torque'] <= spec['peak_torque_10s'] <= spec['peak_torque_3s']:
        raise ValueError(f'{name}: continuous / 10s / 3s torque ordering is invalid')
    dims = spec.get('dimensions', [])
    if len(dims) != 3 or any(not math.isfinite(v) or v <= 0 for v in dims) or dims[0] != dims[1]:
        raise ValueError(f'{name}: expected cylindrical [diameter, diameter, length] in mm')
    curve = spec.get('performance_curve', {})
    keys = ('torque_nm', 'speed_rpm', 'ibus_a', 'efficiency_pct')
    values = [curve.get(k, []) for k in keys]
    if len(values[0]) < 2 or any(len(v) != len(values[0]) for v in values):
        raise ValueError(f'{name}: performance arrays must have matching lengths')
    if any(not math.isfinite(x) or x < 0 for v in values for x in v):
        raise ValueError(f'{name}: performance values must be finite and nonnegative')
    torque, speed, _, efficiency = values
    if (torque[0] != 0 or any(b <= a for a, b in zip(torque, torque[1:]))
            or any(b >= a for a, b in zip(speed, speed[1:]))
            or any(x > 100 for x in efficiency)):
        raise ValueError(f'{name}: invalid monotonic torque/speed or efficiency bounds')
    if torque[-1] != spec['peak_torque_10s'] or speed[0] != spec['no_load_speed_rpm']:
        raise ValueError(f'{name}: curve endpoints do not match the declared 10s/no-load point')
    pairs = [[t, rpm * 2 * math.pi / 60] for t, rpm in zip(torque, speed)]
    return {**spec, 'category': 'MidGear', 'has_dims': True,
            'tau_peak_Nm': spec['peak_torque_10s'], 'omega_NL_rad_s': pairs[0][1],
            'peak_torque': spec['peak_torque_10s'], 'no_load_speed': pairs[0][1],
            'rated_speed': spec.get('rated_speed_rpm', 0) * 2 * math.pi / 60,
            'no_load_speed_datasheet_min': spec.get('no_load_speed_datasheet_min_rpm', 0) * 2 * math.pi / 60,
            'center_bore_diameter_m': spec.get('center_bore_diameter', 0) / 1000,
            'gear': spec['gear_ratio'], 'mass_kg': spec['mass'],
            'OD_mm': dims[0], 'L_mm': dims[2], 'fixed_spec': True,
            'torque_speed_curve': pairs, 'P_peak_W': max(t * w for t, w in pairs)}


def find_by_name(name: str, params: dict | None = None) -> Optional[dict]:
    """Return the catalog entry whose name matches (case/dash/space-insensitive)."""
    norm = _norm(name)
    for entry in ((params or {}).get('_motor_catalog', {}).values()):
        if any(_norm(n) == norm for n in (entry['name'], entry.get('model', ''),
                                         entry['name'].replace('JM', 'M', 1))):
            return entry
    for entry in load_catalog():
        if _norm(entry["name"]) == norm:
            return entry
    return None


def catalog_names() -> list[str]:
    """Return all catalog entry names (for error messages / GUI pickers)."""
    return [e["name"] for e in load_catalog()]


# ── Catalog → parameter dict conversion ───────────────────────────────────────

def catalog_to_class_params(entry: dict) -> dict:
    """Catalogue entry -> r (m), L (m), effort (Nm), velocity (rad/s), rho (kg/m^3).

    rho is mass over the cylindrical package volume.
    """
    r_m = entry["OD_mm"] / 2.0 / 1000.0
    L_m = entry["L_mm"] / 1000.0
    vol = math.pi * r_m**2 * L_m
    rho = entry["mass_kg"] / vol if vol > 0 else 2700.0
    return {
        "r":        r_m,
        "L":        L_m,
        "effort":   entry["tau_peak_Nm"],
        "velocity": entry["omega_NL_rad_s"],
        "rho":      rho,
    }


# ── Named-motor resolution ────────────────────────────────────────────────────

def resolve_catalog_motors(params: dict) -> dict:
    """Fill ``{cls}_motor: "<name>"`` classes with the part's torque, speed and gear.

    Also records the anchor part name. Size and mass are left to the trends;
    keys already in *params* win.
    """
    catalog = load_catalog()
    if not catalog and not params.get('_motor_catalog'):
        return params

    out = dict(params)
    for cls in _declared_classes(params):
        name_val = params.get(f"{cls}_motor")
        if not isinstance(name_val, str):
            continue  # numeric definition — nothing to resolve

        # An explicitly edited/cleared public name supersedes old saved metadata.
        for key in (f"_{cls}_motor_catalog", f"{cls}_motor_catalog_name",
                    f"{cls}_motor_catalog_category"):
            out.pop(key, None)
        if not name_val.strip():
            continue

        entry = find_by_name(name_val, params)
        if entry is None:
            available = ", ".join(catalog_names()[:8]) + " …"
            warnings.warn(
                f"Motor catalog: '{name_val}' (class {cls}) not found in the "
                f"actuator catalogue. Available names include: {available}",
                UserWarning, stacklevel=4,
            )
            continue

        out.setdefault(f"{cls}_motor_effort", entry["tau_peak_Nm"])
        out.setdefault(f"{cls}_motor_velocity", entry["omega_NL_rad_s"])
        gear = gear_ratio_of(entry)
        if gear is not None:
            out.setdefault(f"{cls}_motor_gear", gear)

        out[f"_{cls}_motor_catalog"]         = entry["name"]
        out[f"{cls}_motor_catalog_name"]     = entry["name"]
        out[f"{cls}_motor_catalog_category"] = entry["category"]

    return out


def gear_ratio_of(entry: dict) -> Optional[float]:
    """Reduction ratio of a catalogue entry, from its notes or its name."""
    from .feasibility import parse_gear_ratio
    return parse_gear_ratio(entry.get("notes", "")) or parse_gear_ratio(entry["name"])


def _declared_classes(params: dict) -> list[str]:
    """Motor class letters that appear as ``{cls}_motor*`` keys."""
    seen = []
    for key in params:
        k = str(key)
        for suffix in ("_motor", "_motor_effort"):
            if k.endswith(suffix):
                cls = k[: -len(suffix)]
                if cls and "_" not in cls and cls not in seen:
                    seen.append(cls)
    return seen
