from __future__ import annotations

from datetime import datetime
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from escape_room.algorithms import (
    moving_average,
    run_continuous_policy,
    train_approx_q_learning,
    train_q_learning,
    train_sarsa,
    value_iteration,
)
from escape_room.envs import (
    CONTINUOUS_ACTIONS,
    ContinuousEscapeRoom,
    DynamicObstacleRoom,
    GridEscapeRoom,
    GridState,
    SokobanEscapeRoom,
    continuous_room_config,
    obstacle_room_config,
    room1_config,
    room2_config,
    room3_config,
)
from escape_room.replay import filter_replay_attempts, replay_library_rows


st.set_page_config(page_title="RL Escape Rooms", layout="wide", initial_sidebar_state="collapsed")


ROOM_ORDER = ["dp", "sarsa", "q_learning", "approx", "obstacles"]
ACTION_LABELS = ("Up", "Right", "Down", "Left")

TUNED_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "dp": {"gamma": 0.96, "slip": 0.25, "theta": 1e-4, "max_iterations": 1000, "max_steps": 220},
    "sarsa": {"episodes": 650, "max_steps": 250, "alpha": 0.15, "gamma": 0.96, "epsilon": 0.40, "epsilon_min": 0.03, "epsilon_decay": 0.993, "slip": 0.18},
    "q_learning": {"episodes": 650, "max_steps": 850, "alpha": 0.15, "gamma": 0.96, "epsilon": 0.40, "epsilon_min": 0.03, "epsilon_decay": 0.993, "slip": 0.18},
    "approx": {"episodes": 450, "max_steps": 850, "alpha": 0.08, "gamma": 0.985, "epsilon": 0.40, "epsilon_min": 0.03, "epsilon_decay": 0.993},
    "obstacles": {"episodes": 450, "max_steps": 1400, "alpha": 0.08, "gamma": 0.985, "epsilon": 0.40, "epsilon_min": 0.03, "epsilon_decay": 0.993, "obstacle_count": 7, "observation_range": 3.0},
}

ROOM_THEMES: Dict[str, Dict[str, Any]] = {
    "dp": {
        "menu": "1. Pac-Man Ice Maze",
        "title": "Pac-Man Ice Maze",
        "subtitle": "Classic Pac-Man maze adapted to slippery tiles",
        "algorithm": "Dynamic Programming",
        "inspiration": "Inspired by Pac-Man",
        "art": "pacman-ice-arena.webp",
        "thumbnail_art": "pacman-ice-thumbnail-v2.webp",
        "banner_art": "pacman-ice-banner-v2.webp",
        "mission": "Reach EXIT quickly, avoid moving ghosts and cracks, and handle slippery ice. The complete transition model is known to Value Iteration.",
        "agent": "PAC",
        "goal": "EXIT",
        "state": "(row, col, collected_mask, ghost_phase)",
        "accent": "#38bdf8",
        "agent_color": "#facc15",
        "wall": "#1e3a8a",
        "floor": "#07111f",
        "path": "#123765",
        "danger": "#ef4444",
        "key": "#facc15",
        "portal": "#a855f7",
        "labels": {"start": "PAC", "goal": "EXIT", "trap": "GHOST", "ice": "ICE", "bonus": "PWR"},
        "objectives": ["Reach EXIT", "Avoid moving ghosts", "Handle slippery ice"],
    },
    "sarsa": {
        "menu": "2. Sokoban Vault",
        "title": "Sokoban Vault",
        "subtitle": "Classic crate-pushing puzzle with SARSA",
        "algorithm": "SARSA",
        "inspiration": "Inspired by Sokoban",
        "art": "sokoban-vault-arena.webp",
        "thumbnail_art": "sokoban-vault-thumbnail-v2.webp",
        "banner_art": "sokoban-vault-banner-v2.webp",
        "mission": "Push the BOX onto the target tile, avoid lasers, then enter SAFE. SARSA learns from its actual exploratory moves.",
        "agent": "PUSH",
        "goal": "SAFE",
        "state": "(player_row, player_col, box_row, box_col)",
        "accent": "#a78bfa",
        "agent_color": "#facc15",
        "wall": "#312e81",
        "floor": "#100b24",
        "path": "#2e2363",
        "danger": "#fb7185",
        "key": "#fde047",
        "portal": "#a855f7",
        "labels": {"start": "PUSH", "goal": "SAFE", "trap": "LAS", "ice": "OIL", "bonus": "COIN", "key": "BOX"},
        "objectives": ["Push BOX to target", "Open SAFE", "Learn on-policy"],
    },
    "q_learning": {
        "menu": "3. Bomberman Reactor",
        "title": "Bomberman Reactor",
        "subtitle": "Bomberman-style grid with bombs and warp tunnels",
        "algorithm": "Q-Learning",
        "inspiration": "Inspired by Bomberman",
        "art": "bomberman-reactor-arena.webp",
        "thumbnail_art": "bomberman-reactor-thumbnail-v2.webp",
        "banner_art": "bomberman-reactor-banner-v2.webp",
        "mission": "Collect two CORE items, avoid bomb blasts and patrol bots, use WARP tunnels, then escape through GATE.",
        "agent": "BOMB",
        "goal": "GATE",
        "state": "(row, col, collected_cores, guard_phase)",
        "accent": "#2dd4bf",
        "agent_color": "#facc15",
        "wall": "#134e4a",
        "floor": "#041411",
        "path": "#123d37",
        "danger": "#f97316",
        "key": "#fde047",
        "portal": "#c084fc",
        "labels": {"start": "START", "goal": "GATE", "trap": "BOMB", "ice": "SLIME", "bonus": "CHG", "key": "CORE", "portal": "WARP", "guard": "BOT"},
        "objectives": ["Collect 2 COREs", "Avoid bombs", "Use WARP"],
    },
    "approx": {
        "menu": "4. Lunar Lander Pad",
        "title": "Lunar Lander Pad",
        "subtitle": "Atari Lunar Lander-style continuous landing",
        "algorithm": "Approximate Q-Learning",
        "inspiration": "Inspired by Lunar Lander",
        "art": "lunar-lander-arena.webp",
        "thumbnail_art": "lunar-lander-thumbnail-v2.webp",
        "banner_art": "lunar-lander-banner-v2.webp",
        "mission": "Choose a discrete velocity every 0.02 seconds, avoid asteroid fields, and guide the lander across a continuous 10x10 meter room to PAD.",
        "agent": "LANDER",
        "goal": "PAD",
        "state": "X, Y, Vx, Vy",
        "accent": "#60a5fa",
        "agent_color": "#facc15",
        "wall": "#1d4ed8",
        "floor": "#061323",
        "path": "#1d3557",
        "danger": "#ef4444",
        "key": "#fde047",
        "portal": "#a855f7",
        "objectives": ["Avoid asteroid fields", "Minimize time", "Touch down on PAD"],
    },
    "obstacles": {
        "menu": "5. Portal Hazard Run",
        "title": "Portal Hazard Run",
        "subtitle": "Moving portal hazards with local forward observation",
        "algorithm": "Approximate Q-Learning",
        "inspiration": "Inspired by Portal",
        "art": "portal-arena.webp",
        "thumbnail_art": "portal-hazard-thumbnail-v2.webp",
        "banner_art": "portal-hazard-banner-v2.webp",
        "mission": "Avoid moving portal hazards and reach EXIT. The agent observes only the nearest portal within X meters in front of it.",
        "agent": "PORTAL",
        "goal": "EXIT",
        "state": "X, Y, Vx, Vy, obstacle_dx, obstacle_dy, visible",
        "accent": "#fb923c",
        "agent_color": "#facc15",
        "wall": "#7c2d12",
        "floor": "#170d05",
        "path": "#4a250c",
        "danger": "#dc2626",
        "key": "#fde047",
        "portal": "#a855f7",
        "objectives": ["Observe forward", "Avoid moving portals", "Reach EXIT"],
    },
}

ROOMS = {ROOM_THEMES[kind]["menu"]: kind for kind in ROOM_ORDER}


def css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #080a0d;
            --panel: #101319;
            --panel2: #171b22;
            --panel3: #20252e;
            --text: #f4f7fb;
            --muted: #a5adb9;
            --line: #303640;
            --focus: #67e8f9;
            --signal: #facc15;
            --success: #4ade80;
            --danger-ui: #fb7185;
        }
        .stApp {
            background:
                linear-gradient(rgba(255,255,255,.018) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,.018) 1px, transparent 1px),
                #080a0d;
            background-size: 36px 36px;
            color: var(--text);
        }
        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        #MainMenu,
        footer {
            visibility: hidden;
            height: 0;
        }
        .stDeployButton {
            display: none;
        }
        .block-container {
            padding-top: .5rem;
            padding-bottom: 3rem;
            max-width: 1440px;
        }
        [data-testid="stSidebar"] {
            background: #0d1015;
            border-left: 1px solid var(--line);
        }
        h1, h2, h3, p, label, span {
            letter-spacing: 0;
        }
        .topbar {
            position: sticky;
            top: .5rem;
            z-index: 50;
            border: 1px solid #343b46;
            border-left: 4px solid var(--signal);
            background: rgba(12, 15, 20, .94);
            backdrop-filter: blur(18px);
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 14px;
            box-shadow: 0 14px 38px rgba(0, 0, 0, .38);
        }
        .topbar-inner {
            display: flex;
            justify-content: space-between;
            gap: 20px;
            align-items: center;
        }
        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 11px;
            min-width: 0;
        }
        .brand-mark {
            display: grid;
            place-items: center;
            width: 42px;
            height: 42px;
            flex: 0 0 42px;
            border: 1px solid #59616d;
            border-radius: 7px;
            background: var(--signal);
            color: #090b0f;
            font-size: .9rem;
            font-weight: 950;
            box-shadow: 4px 4px 0 #303640;
        }
        .topbar h1 {
            margin: 0;
            color: #ffffff;
            font-size: 1.15rem;
            line-height: 1.1;
            font-weight: 950;
        }
        .topbar p {
            margin: 4px 0 0;
            color: #9ca3af;
            font-size: .78rem;
            font-weight: 750;
            text-transform: uppercase;
        }
        .topbar-badges {
            display: flex;
            gap: 8px;
            align-items: stretch;
            justify-content: flex-end;
        }
        .topbar-badge {
            display: grid;
            gap: 2px;
            min-width: 106px;
            border-left: 1px solid #353b45;
            padding: 3px 12px;
            color: #a5adb9;
            font-weight: 850;
            font-size: .68rem;
            white-space: nowrap;
            text-transform: uppercase;
        }
        .topbar-badge strong {
            overflow: hidden;
            color: #f4f7fb;
            font-size: .84rem;
            text-overflow: ellipsis;
            text-transform: none;
        }
        .topbar-badge.active strong {
            color: var(--focus);
        }
        .status-live {
            display: inline-block;
            width: 7px;
            height: 7px;
            margin-right: 5px;
            border-radius: 50%;
            background: var(--success);
            box-shadow: 0 0 9px rgba(74,222,128,.7);
        }
        div[data-testid="stTabs"] {
            border: 1px solid var(--line);
            background: rgba(13, 16, 21, .86);
            border-radius: 8px;
            padding: 8px 10px 12px;
            margin-top: 10px;
        }
        div[data-testid="stTabs"] [role="tablist"] {
            display: flex;
            width: 100%;
            gap: 8px;
            border-bottom: 1px solid var(--line);
            padding-bottom: 8px;
        }
        div[data-testid="stTabs"] button[role="tab"] {
            flex: 1 1 0;
            justify-content: center;
            min-height: 42px;
            border-radius: 6px;
            border: 1px solid transparent;
            background: #171b22;
            color: #c6ccd5 !important;
            font-weight: 900;
            padding: 0 12px;
            opacity: 1 !important;
        }
        div[data-testid="stTabs"] button[role="tab"] * {
            color: #c6ccd5 !important;
            font-weight: 900;
            opacity: 1 !important;
        }
        div[data-testid="stTabs"] button[role="tab"]:hover {
            border-color: #46505d;
            background: #20252e;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            border-color: var(--signal);
            background: var(--signal);
            color: #090b0f !important;
            box-shadow: none;
        }
        div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] * {
            color: #090b0f !important;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: var(--signal) !important;
        }
        .game-card {
            --accent: #38bdf8;
            position: relative;
            min-height: 300px;
            overflow: hidden;
            border: 1px solid color-mix(in srgb, var(--accent), #303640 64%);
            border-radius: 8px;
            padding: 22px 24px;
            margin-bottom: 12px;
            background-color: #11151b;
            background-position: center;
            background-size: cover;
            box-shadow: 0 18px 46px rgba(0,0,0,.34);
        }
        .game-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                linear-gradient(90deg, rgba(5,7,10,.97) 0%, rgba(5,7,10,.88) 40%, rgba(5,7,10,.38) 68%, rgba(5,7,10,.12) 100%),
                linear-gradient(180deg, rgba(5,7,10,.08), rgba(5,7,10,.30));
        }
        .game-card > * {
            position: relative;
            z-index: 1;
        }
        .mission-kicker {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #c8ced7;
            font-size: .72rem;
            font-weight: 900;
            text-transform: uppercase;
        }
        .mission-kicker span {
            padding: 4px 7px;
            border: 1px solid #49515d;
            border-radius: 4px;
            background: rgba(9,11,15,.72);
            white-space: nowrap;
        }
        .mission-kicker .room-index {
            border-color: var(--accent);
            color: var(--accent);
        }
        .game-card h2 {
            max-width: 720px;
            color: #ffffff;
            margin: 16px 0 0;
            font-size: 2rem;
            line-height: 1.05;
            font-weight: 950;
        }
        .game-card .sub {
            color: var(--accent);
            font-weight: 900;
            margin-top: 7px;
        }
        .game-card .mission {
            max-width: 760px;
            color: #d7dce4;
            margin-top: 10px;
            line-height: 1.5;
        }
        .objective-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 16px;
        }
        .objective {
            border-left: 3px solid var(--accent);
            background: rgba(9,11,15,.78);
            border-radius: 4px;
            padding: 8px 12px;
            color: #eef2f7;
            font-weight: 800;
        }
        .hud {
            --accent: #38bdf8;
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 12px 0;
        }
        .hud-card {
            border: 1px solid var(--line);
            border-top: 3px solid #47505d;
            background: #14181f;
            border-radius: 6px;
            padding: 10px 12px;
            box-shadow: 0 10px 24px rgba(0,0,0,.2);
        }
        .hud-label {
            color: #94a3b8;
            font-size: .76rem;
            font-weight: 800;
            text-transform: uppercase;
        }
        .hud-value {
            color: #f8fafc;
            font-size: 1.06rem;
            font-weight: 900;
            margin-top: 3px;
        }
        .hud-card.highlight .hud-value {
            color: var(--accent);
        }
        .hud-card.highlight {
            border-top-color: var(--accent);
        }
        .arcade-shell {
            --accent: #38bdf8;
            --wall: #1e3a8a;
            --floor: #07111f;
            --path: #123765;
            --danger: #ef4444;
            --key: #facc15;
            --portal: #a855f7;
            --agent: #facc15;
            direction: ltr;
            width: min(100%, 650px);
            margin: 0 auto;
            border-radius: 18px;
            padding: 16px;
            border: 2px solid var(--accent);
            background: #020617;
            box-shadow: 0 0 0 4px rgba(255,255,255,.03), 0 0 28px color-mix(in srgb, var(--accent), transparent 55%);
        }
        .arcade-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #f8fafc;
            font-weight: 900;
            margin-bottom: 10px;
            font-size: .88rem;
        }
        .maze {
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            gap: 4px;
            background: #020617;
        }
        .tile {
            position: relative;
            aspect-ratio: 1 / 1;
            border-radius: 7px;
            background: var(--floor);
            box-shadow: inset 0 0 0 1px rgba(148, 163, 184, .12);
            overflow: hidden;
        }
        .tile::after {
            content: "";
            position: absolute;
            left: 50%;
            top: 50%;
            width: 6px;
            height: 6px;
            margin-left: -3px;
            margin-top: -3px;
            border-radius: 999px;
            background: rgba(248, 250, 252, .58);
            box-shadow: 0 0 8px rgba(248,250,252,.35);
        }
        .tile.visited { background: var(--path); }
        .tile.wall {
            background: linear-gradient(135deg, var(--wall), #172554);
            box-shadow: inset 0 0 0 2px color-mix(in srgb, var(--accent), white 10%), 0 0 12px color-mix(in srgb, var(--accent), transparent 75%);
        }
        .tile.wall::after, .tile.agent::after, .tile.goal::after, .tile.key::after,
        .tile.trap::after, .tile.slippery::after, .tile.portal::after, .tile.guard::after,
        .tile.bonus::after {
            display: none;
        }
        .tile.slippery {
            background: repeating-linear-gradient(135deg, #bae6fd 0 6px, #e0f2fe 6px 12px);
        }
        .tile.trap {
            background: #1f2937;
        }
        .tile.trap::before {
            content: "";
            position: absolute;
            left: 8%;
            right: 8%;
            top: 45%;
            height: 5px;
            background: var(--danger);
            border-radius: 999px;
            box-shadow: 0 0 12px var(--danger);
            transform: rotate(-32deg);
        }
        .tile.goal {
            background: radial-gradient(circle, #22c55e, #065f46);
            box-shadow: 0 0 16px rgba(34,197,94,.6), inset 0 0 0 2px #bbf7d0;
        }
        .tile.goal .label, .tile.key .label, .tile.portal .label, .tile.bonus .label,
        .tile.guard .label, .tile.start .label {
            position: absolute;
            inset: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #f8fafc;
            font-size: clamp(8px, 1vw, 12px);
            font-weight: 900;
            text-align: center;
        }
        .tile.key {
            background: radial-gradient(circle, var(--key), #a16207);
            box-shadow: 0 0 16px rgba(250,204,21,.55);
        }
        .tile.key .label { color: #111827; }
        .tile.portal {
            background: conic-gradient(from 0deg, #4c1d95, var(--portal), #22d3ee, #4c1d95);
            box-shadow: 0 0 18px rgba(168,85,247,.55);
        }
        .tile.guard {
            background: #7c2d12;
            box-shadow: 0 0 16px rgba(249,115,22,.55);
        }
        .tile.guard .label { color: #fed7aa; }
        .tile.bonus {
            background: radial-gradient(circle, #fef3c7, #f59e0b);
        }
        .tile.bonus .label { color: #111827; }
        .tile.start {
            background: #111827;
            box-shadow: inset 0 0 0 2px rgba(148,163,184,.45);
        }
        .tile.agent {
            background: var(--floor);
        }
        .pac {
            position: absolute;
            inset: 14%;
            border-radius: 999px;
            background: conic-gradient(from 35deg, transparent 0 70deg, var(--agent) 70deg 360deg);
            filter: drop-shadow(0 0 9px rgba(250,204,21,.65));
            animation: chomp .42s linear infinite alternate;
        }
        @keyframes chomp {
            from { clip-path: polygon(100% 50%, 100% 50%, 100% 50%, 100% 50%, 100% 50%, 50% 50%, 0 0, 0 100%); }
            to { clip-path: polygon(100% 25%, 100% 25%, 100% 75%, 100% 75%, 100% 75%, 50% 50%, 0 0, 0 100%); }
        }
        .label {
            letter-spacing: 0;
        }
        .arena {
            --accent: #60a5fa;
            --floor: #061323;
            --danger: #ef4444;
            --agent: #facc15;
            position: relative;
            width: min(100%, 650px);
            aspect-ratio: 1 / 1;
            margin: 0 auto;
            border-radius: 18px;
            border: 2px solid var(--accent);
            overflow: hidden;
            background:
                linear-gradient(rgba(148,163,184,.12) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148,163,184,.12) 1px, transparent 1px),
                var(--floor);
            background-size: 10% 10%;
            box-shadow: 0 0 28px color-mix(in srgb, var(--accent), transparent 55%);
        }
        .arena-path {
            position: absolute;
            width: 4px;
            height: 4px;
            border-radius: 999px;
            background: rgba(255,255,255,.7);
            box-shadow: 0 0 9px var(--accent);
        }
        .arena-hazard {
            position: absolute;
            border-radius: 8px;
            background: rgba(239, 68, 68, .42);
            border: 1px solid rgba(248,113,113,.75);
            box-shadow: 0 0 18px rgba(239,68,68,.45);
        }
        .arena-obstacle {
            position: absolute;
            border-radius: 6px;
            background: #f97316;
            border: 2px solid #fed7aa;
            box-shadow: 0 0 16px rgba(249,115,22,.55);
        }
        .arena-goal {
            position: absolute;
            border-radius: 999px;
            background: rgba(34, 197, 94, .33);
            border: 2px solid #86efac;
            box-shadow: 0 0 22px rgba(34,197,94,.65);
            transform: translate(-50%, -50%);
        }
        .arena-agent {
            position: absolute;
            width: 20px;
            height: 20px;
            border-radius: 999px;
            background: var(--agent);
            transform: translate(-50%, -50%);
            box-shadow: 0 0 16px rgba(250,204,21,.75);
        }
        .control-pad {
            display: grid;
            grid-template-columns: repeat(3, minmax(80px, 1fr));
            gap: 8px;
            max-width: 330px;
            margin: 0 auto;
        }
        .small-note {
            color: #94a3b8;
            font-size: .9rem;
            line-height: 1.45;
        }
        div.stButton > button,
        div.stDownloadButton > button,
        div.stFormSubmitButton > button,
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-primary"],
        button[data-testid="stFormSubmitButton"] {
            min-height: 44px;
            border-radius: 6px;
            border: 1px solid #4a535f !important;
            background: #20252d !important;
            color: #f8fafc !important;
            font-weight: 900 !important;
            box-shadow: 0 8px 18px rgba(0, 0, 0, .22);
        }
        div.stButton > button *,
        div.stDownloadButton > button *,
        div.stFormSubmitButton > button *,
        button[data-testid="stBaseButton-secondary"] *,
        button[data-testid="stBaseButton-primary"] *,
        button[data-testid="stFormSubmitButton"] * {
            color: #f8fafc !important;
            font-weight: 900 !important;
            opacity: 1 !important;
        }
        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        div.stFormSubmitButton > button:hover,
        button[data-testid="stBaseButton-secondary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover,
        button[data-testid="stFormSubmitButton"]:hover {
            border-color: var(--signal) !important;
            background: var(--signal) !important;
            color: #090b0f !important;
            box-shadow: 0 10px 24px rgba(0,0,0,.32);
            transform: translateY(-1px);
        }
        div.stButton > button:hover *,
        div.stDownloadButton > button:hover *,
        div.stFormSubmitButton > button:hover *,
        button[data-testid="stBaseButton-secondary"]:hover *,
        button[data-testid="stBaseButton-primary"]:hover *,
        button[data-testid="stFormSubmitButton"]:hover * {
            color: #090b0f !important;
        }
        div.stButton > button:focus,
        div.stDownloadButton > button:focus,
        div.stFormSubmitButton > button:focus,
        button[data-testid="stBaseButton-secondary"]:focus,
        button[data-testid="stBaseButton-primary"]:focus,
        button[data-testid="stFormSubmitButton"]:focus {
            outline: 3px solid rgba(103, 232, 249, .45) !important;
            outline-offset: 2px;
        }
        div.stButton > button:disabled,
        button[data-testid="stBaseButton-secondary"]:disabled {
            border-color: #313741 !important;
            background: #12161b !important;
            color: #69727e !important;
            box-shadow: none !important;
            transform: none !important;
            cursor: not-allowed;
        }
        div.stButton > button:disabled *,
        button[data-testid="stBaseButton-secondary"]:disabled * {
            color: #69727e !important;
        }
        .select-head {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 20px;
            align-items: end;
            border-top: 1px solid #353c46;
            border-bottom: 1px solid #353c46;
            padding: 22px 2px 20px;
            margin-bottom: 14px;
        }
        .select-head h2 {
            margin: 0;
            color: #ffffff;
            font-size: 2rem;
            line-height: 1.08;
            font-weight: 950;
        }
        .select-head p {
            margin: 7px 0 0 0;
            color: #aeb6c1;
            max-width: 760px;
        }
        .select-eyebrow {
            margin-bottom: 7px;
            color: var(--signal);
            font-size: .72rem;
            font-weight: 950;
            text-transform: uppercase;
        }
        .campaign-score {
            min-width: 148px;
            text-align: right;
        }
        .campaign-score strong {
            display: block;
            color: #ffffff;
            font-size: 2.5rem;
            line-height: 1;
        }
        .campaign-score span {
            color: #9ca3af;
            font-size: .72rem;
            font-weight: 900;
            text-transform: uppercase;
        }
        .campaign-track {
            position: relative;
            height: 8px;
            overflow: hidden;
            border: 1px solid #3a414c;
            border-radius: 4px;
            background: #171b22;
            margin: 0 0 18px;
        }
        .campaign-track span {
            display: block;
            width: var(--progress);
            height: 100%;
            background: linear-gradient(90deg, var(--focus), var(--success));
        }
        .room-select-card {
            --accent: #38bdf8;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            height: 510px;
            min-height: 510px;
            border: 1px solid #303741;
            border-top: 3px solid var(--accent);
            background: #12161c;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            box-shadow: 0 16px 40px rgba(0,0,0,.28);
            transition: transform .2s ease, border-color .2s ease, box-shadow .2s ease;
        }
        .room-select-card:hover {
            transform: translateY(-2px);
            border-color: var(--accent);
            box-shadow: 0 22px 50px rgba(0,0,0,.38);
        }
        .room-select-card .screen {
            flex: 0 0 160px;
            height: 160px;
            border-radius: 6px;
            border: 1px solid color-mix(in srgb, var(--accent), white 15%);
            background:
                linear-gradient(rgba(148,163,184,.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148,163,184,.10) 1px, transparent 1px),
                #020617;
            background-size: 12.5% 25%;
            position: relative;
            overflow: hidden;
            margin-bottom: 14px;
            box-shadow: inset 0 0 34px rgba(0,0,0,.5);
        }
        .room-select-card .screen::before,
        .room-select-card .screen::after {
            position: absolute;
        }
        .thumb-name {
            position: absolute;
            z-index: 4;
            left: 12px;
            top: 10px;
            padding: 4px 8px;
            border-radius: 999px;
            border: 1px solid color-mix(in srgb, var(--accent), white 30%);
            background: rgba(2, 6, 23, .78);
            color: #ffffff;
            font-size: .72rem;
            font-weight: 900;
            letter-spacing: 0;
            box-shadow: 0 0 16px color-mix(in srgb, var(--accent), transparent 70%);
        }
        .mini-wall,
        .mini-wall::before,
        .mini-wall::after,
        .mini-crate,
        .mini-target,
        .mini-laser,
        .mini-bomb,
        .mini-blast,
        .mini-meteor,
        .mini-pad,
        .mini-portal,
        .mini-route {
            position: absolute;
        }
        .mini-wall {
            border-radius: 7px;
            background: color-mix(in srgb, var(--accent), #0f172a 58%);
            border: 2px solid color-mix(in srgb, var(--accent), white 15%);
            box-shadow: 0 0 18px color-mix(in srgb, var(--accent), transparent 70%);
        }
        .mini-pellet {
            position: absolute;
            width: 7px;
            height: 7px;
            border-radius: 999px;
            background: #f8fafc;
            box-shadow: 0 0 10px rgba(248,250,252,.75);
        }
        .mini-pac {
            position: absolute;
            width: 31px;
            height: 31px;
            border-radius: 999px;
            background: conic-gradient(from 34deg, transparent 0 70deg, #facc15 70deg 360deg);
            filter: drop-shadow(0 0 12px rgba(250,204,21,.8));
        }
        .mini-ghost {
            position: absolute;
            width: 30px;
            height: 30px;
            border-radius: 16px 16px 7px 7px;
            background: #fb7185;
            box-shadow: 0 0 16px rgba(251,113,133,.68);
        }
        .mini-ghost::before {
            content: "";
            position: absolute;
            left: 7px;
            top: 8px;
            width: 5px;
            height: 5px;
            border-radius: 50%;
            background: #fff;
            box-shadow: 12px 0 #fff;
        }
        .mini-ghost::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: -1px;
            height: 8px;
            background: repeating-linear-gradient(90deg, #fb7185 0 7px, transparent 7px 13px);
        }
        .preview-dp .w1 { left: 14%; top: 31%; width: 62%; height: 13px; }
        .preview-dp .w2 { left: 14%; top: 56%; width: 44%; height: 13px; }
        .preview-dp .w3 { right: 11%; top: 19%; width: 12px; height: 61%; }
        .preview-dp .p1 { left: 18%; top: 73%; }
        .preview-dp .p2 { left: 31%; top: 73%; }
        .preview-dp .p3 { left: 45%; top: 73%; }
        .preview-dp .p4 { left: 59%; top: 73%; }
        .preview-dp .mini-pac { left: 22%; top: 39%; }
        .preview-dp .mini-ghost { right: 18%; top: 54%; }

        .preview-sarsa {
            background:
                linear-gradient(rgba(233,213,255,.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(233,213,255,.10) 1px, transparent 1px),
                #130d25;
            background-size: 12.5% 25%;
        }
        .mini-crate {
            width: 34px;
            height: 34px;
            border-radius: 6px;
            background:
                linear-gradient(45deg, transparent 44%, rgba(120,53,15,.75) 45% 55%, transparent 56%),
                linear-gradient(-45deg, transparent 44%, rgba(120,53,15,.75) 45% 55%, transparent 56%),
                #d97706;
            border: 2px solid #fde68a;
            box-shadow: 0 0 18px rgba(251,191,36,.42);
        }
        .mini-target {
            width: 42px;
            height: 42px;
            border-radius: 9px;
            border: 3px dashed #86efac;
            background: rgba(34,197,94,.12);
            box-shadow: 0 0 18px rgba(134,239,172,.35);
        }
        .mini-player {
            position: absolute;
            width: 26px;
            height: 34px;
            border-radius: 12px 12px 8px 8px;
            background: #facc15;
            border: 2px solid #111827;
            box-shadow: 0 0 14px rgba(250,204,21,.55);
        }
        .mini-player::before {
            content: "";
            position: absolute;
            left: 4px;
            top: 5px;
            width: 16px;
            height: 5px;
            border-radius: 999px;
            background: #111827;
        }
        .mini-laser {
            height: 4px;
            border-radius: 999px;
            background: #fb7185;
            box-shadow: 0 0 14px rgba(251,113,133,.75);
            transform: rotate(-8deg);
        }
        .preview-sarsa .mini-crate { left: 37%; top: 43%; }
        .preview-sarsa .mini-target { right: 13%; top: 38%; }
        .preview-sarsa .mini-player { left: 18%; top: 42%; }
        .preview-sarsa .l1 { left: 20%; right: 14%; top: 29%; }
        .preview-sarsa .l2 { left: 12%; right: 24%; top: 73%; transform: rotate(5deg); }

        .preview-q_learning {
            background:
                radial-gradient(circle at 72% 32%, rgba(249,115,22,.22), transparent 12%),
                linear-gradient(rgba(45,212,191,.10) 1px, transparent 1px),
                linear-gradient(90deg, rgba(45,212,191,.10) 1px, transparent 1px),
                #041411;
            background-size: auto, 12.5% 25%, 12.5% 25%;
        }
        .mini-bomber {
            position: absolute;
            width: 31px;
            height: 31px;
            border-radius: 999px;
            background: #facc15;
            border: 3px solid #0f172a;
            box-shadow: 0 0 14px rgba(250,204,21,.65);
        }
        .mini-bomber::before {
            content: "";
            position: absolute;
            left: 5px;
            top: 7px;
            width: 15px;
            height: 5px;
            background: #0f172a;
            box-shadow: 0 9px #0f172a;
        }
        .mini-bomb {
            width: 27px;
            height: 27px;
            border-radius: 999px;
            background: #111827;
            border: 2px solid #fed7aa;
            box-shadow: 0 0 16px rgba(249,115,22,.8);
        }
        .mini-bomb::before {
            content: "";
            position: absolute;
            right: -6px;
            top: -7px;
            width: 13px;
            height: 13px;
            border-radius: 50%;
            background: #f97316;
        }
        .mini-blast {
            width: 76px;
            height: 9px;
            border-radius: 999px;
            background: #fb923c;
            box-shadow: 0 0 16px rgba(251,146,60,.75);
        }
        .mini-blast::before {
            content: "";
            position: absolute;
            left: 34px;
            top: -31px;
            width: 9px;
            height: 72px;
            border-radius: 999px;
            background: #fb923c;
        }
        .preview-q_learning .w1 { left: 19%; top: 27%; width: 13px; height: 56%; }
        .preview-q_learning .w2 { left: 52%; top: 18%; width: 13px; height: 65%; }
        .preview-q_learning .mini-bomber { left: 25%; top: 51%; }
        .preview-q_learning .mini-bomb { right: 20%; top: 38%; }
        .preview-q_learning .mini-blast { right: 12%; top: 62%; }

        .preview-approx {
            background:
                radial-gradient(circle at 20% 22%, rgba(248,250,252,.65) 0 2px, transparent 3px),
                radial-gradient(circle at 69% 18%, rgba(248,250,252,.55) 0 2px, transparent 3px),
                radial-gradient(circle at 42% 43%, rgba(248,250,252,.45) 0 1px, transparent 2px),
                linear-gradient(180deg, #020617 0%, #10223d 100%);
        }
        .mini-lander {
            position: absolute;
            left: 30%;
            top: 20%;
            width: 52px;
            height: 44px;
        }
        .mini-lander::before {
            content: "";
            position: absolute;
            left: 13px;
            top: 0;
            width: 26px;
            height: 30px;
            clip-path: polygon(50% 0, 100% 100%, 0 100%);
            background: #e0f2fe;
            border: 2px solid #60a5fa;
            filter: drop-shadow(0 0 12px rgba(96,165,250,.65));
        }
        .mini-lander::after {
            content: "";
            position: absolute;
            left: 20px;
            top: 31px;
            width: 12px;
            height: 21px;
            clip-path: polygon(50% 100%, 0 0, 100% 0);
            background: #fb923c;
            filter: drop-shadow(0 0 12px rgba(251,146,60,.8));
        }
        .mini-pad {
            right: 14%;
            bottom: 17%;
            width: 86px;
            height: 12px;
            border-radius: 999px;
            background: #86efac;
            box-shadow: 0 0 18px rgba(134,239,172,.65);
        }
        .mini-moon {
            position: absolute;
            left: 0;
            right: 0;
            bottom: 0;
            height: 34px;
            background: linear-gradient(135deg, #475569, #94a3b8);
            clip-path: polygon(0 65%, 11% 43%, 24% 61%, 36% 28%, 51% 64%, 65% 35%, 80% 58%, 100% 34%, 100% 100%, 0 100%);
        }
        .mini-meteor {
            right: 33%;
            top: 28%;
            width: 24px;
            height: 24px;
            border-radius: 999px;
            background: #ef4444;
            box-shadow: -17px 10px 0 rgba(239,68,68,.35), 0 0 16px rgba(239,68,68,.75);
        }

        .preview-obstacles {
            background:
                radial-gradient(circle at 25% 33%, rgba(34,211,238,.22), transparent 13%),
                radial-gradient(circle at 70% 60%, rgba(168,85,247,.24), transparent 16%),
                linear-gradient(135deg, #170d05 0%, #261044 100%);
        }
        .mini-portal {
            width: 58px;
            height: 58px;
            border-radius: 999px;
            border: 5px solid #22d3ee;
            box-shadow: 0 0 22px rgba(34,211,238,.75), inset 0 0 18px rgba(168,85,247,.55);
        }
        .mini-portal.purple {
            border-color: #f0abfc;
            box-shadow: 0 0 22px rgba(240,171,252,.75), inset 0 0 18px rgba(34,211,238,.55);
        }
        .mini-runner {
            position: absolute;
            left: 46%;
            top: 47%;
            width: 25px;
            height: 35px;
        }
        .mini-runner::before {
            content: "";
            position: absolute;
            left: 7px;
            top: 0;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #facc15;
            box-shadow: 0 0 12px rgba(250,204,21,.7);
        }
        .mini-runner::after {
            content: "";
            position: absolute;
            left: 11px;
            top: 13px;
            width: 4px;
            height: 23px;
            border-radius: 999px;
            background: #facc15;
            box-shadow: -9px 8px 0 -1px #facc15, 10px 9px 0 -1px #facc15;
        }
        .mini-route {
            left: 25%;
            right: 24%;
            top: 58%;
            height: 3px;
            border-top: 3px dashed #fed7aa;
            transform: rotate(-10deg);
            opacity: .85;
        }
        .preview-obstacles .portal-a { left: 16%; top: 32%; }
        .preview-obstacles .portal-b { right: 14%; top: 30%; }
        .room-card-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 9px;
            color: #aab2bd;
            font-size: .7rem;
            font-weight: 900;
            text-transform: uppercase;
        }
        .room-number {
            color: var(--accent);
        }
        .room-status {
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }
        .room-status::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #6b7280;
        }
        .room-status.ready::before {
            background: var(--success);
            box-shadow: 0 0 8px rgba(74,222,128,.6);
        }
        .room-status.completed::before {
            background: var(--signal);
            box-shadow: 0 0 8px rgba(250,204,21,.6);
        }
        .room-select-card h3 {
            color: #ffffff;
            font-size: 1.24rem;
            line-height: 1.15;
            margin: 0;
            font-weight: 950;
        }
        .room-select-card .classic {
            color: #9fa7b2;
            font-size: .74rem;
            font-weight: 800;
            margin-top: 5px;
        }
        .room-select-card .algo {
            color: var(--accent);
            font-size: .78rem;
            font-weight: 900;
            margin-top: 12px;
            text-transform: uppercase;
        }
        .room-state {
            color: #c8ced7;
            font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
            font-size: .72rem;
            margin-top: 5px;
        }
        .room-select-card p {
            color: #bbc2cc;
            font-size: .84rem;
            line-height: 1.5;
            margin: 9px 0 0 0;
        }
        .room-objectives {
            display: grid;
            gap: 5px;
            margin-top: auto;
            padding-top: 12px;
        }
        .room-objectives span {
            display: flex;
            align-items: center;
            gap: 7px;
            color: #d7dce4;
            font-size: .76rem;
            font-weight: 800;
        }
        .room-objectives span::before {
            content: "";
            width: 6px;
            height: 6px;
            flex: 0 0 6px;
            border: 1px solid var(--accent);
            background: color-mix(in srgb, var(--accent), transparent 70%);
        }
        .room-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
            min-height: 44px;
            border-top: 1px solid #353c46;
            border-bottom: 1px solid #353c46;
            padding: 8px 4px;
            margin-bottom: 10px;
        }
        .room-nav .current {
            color: #f8fafc;
            font-weight: 900;
        }
        .room-nav .hint {
            color: #94a3b8;
            font-size: .78rem;
        }
        .room-nav .crumb {
            color: var(--accent);
            font-size: .7rem;
            font-weight: 900;
            text-transform: uppercase;
        }
        div[data-testid="stTabs"] button {
            font-weight: 900;
        }
        [data-testid="stWidgetLabel"] {
            margin-bottom: 6px;
        }
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label,
        [data-testid="stWidgetLabel"] span {
            color: #f8fafc !important;
            font-weight: 900 !important;
            font-size: .96rem !important;
            opacity: 1 !important;
        }
        [data-testid="stRadio"] label,
        [data-testid="stRadio"] label p,
        [data-testid="stRadio"] label span {
            color: #e5e7eb !important;
            font-weight: 800 !important;
            opacity: 1 !important;
        }
        [data-testid="stForm"] {
            border: 1px solid var(--line);
            background: #101319;
            border-radius: 8px;
            padding: 18px;
            box-shadow: 0 16px 34px rgba(0,0,0,.24);
        }
        .train-guide {
            border-left: 4px solid var(--focus);
            background: #14181f;
            border-radius: 6px;
            padding: 14px 16px;
            margin: 8px 0 14px;
        }
        .train-guide h3 {
            margin: 0 0 8px 0;
            color: #f8fafc;
            font-size: 1.2rem;
        }
        .train-guide p {
            margin: 0;
            color: #cbd5e1;
            line-height: 1.45;
        }
        .param-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin-top: 12px;
        }
        .param-card {
            border-top: 1px solid #3d4551;
            background: #191e26;
            border-radius: 5px;
            padding: 10px 11px;
            color: #dbeafe;
            min-height: 76px;
        }
        .param-card b {
            display: block;
            color: #ffffff;
            margin-bottom: 4px;
            font-size: .93rem;
        }
        .param-card span {
            color: #cbd5e1 !important;
            font-size: .84rem;
            line-height: 1.35;
        }
        .form-section {
            margin: 0 0 10px 0;
            padding: 8px 10px;
            border-radius: 5px;
            border-left: 3px solid var(--signal);
            background: #1a1f27;
            color: #ffffff;
            font-weight: 900;
        }
        .section-head {
            display: flex;
            justify-content: space-between;
            gap: 18px;
            align-items: end;
            padding: 12px 2px 14px;
            margin-bottom: 8px;
            border-bottom: 1px solid #343b45;
        }
        .section-kicker {
            color: var(--focus);
            font-size: .7rem;
            font-weight: 950;
            text-transform: uppercase;
        }
        .section-head h2 {
            margin: 4px 0 0;
            color: #ffffff;
            font-size: 1.45rem;
            font-weight: 950;
        }
        .section-head p {
            max-width: 650px;
            margin: 0;
            color: #aeb6c1;
            font-size: .86rem;
            line-height: 1.45;
            text-align: right;
        }
        .empty-state {
            display: grid;
            grid-template-columns: 44px minmax(0, 1fr);
            gap: 12px;
            align-items: center;
            border: 1px dashed #46505c;
            border-radius: 7px;
            padding: 18px;
            background: #12161c;
            color: #dce1e8;
        }
        .empty-state strong {
            display: block;
            color: #ffffff;
            margin-bottom: 3px;
        }
        .empty-state span {
            color: #aeb6c1;
            font-size: .86rem;
        }
        .empty-icon {
            display: grid;
            place-items: center;
            width: 44px;
            height: 44px;
            border: 1px solid #49525f;
            border-radius: 6px;
            color: var(--signal);
            font-size: 1.2rem;
            font-weight: 950;
        }
        .details-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 9px;
            margin: 10px 0 14px;
        }
        .detail-card {
            border-top: 3px solid var(--accent);
            background: #151920;
            border-radius: 6px;
            padding: 12px;
            min-height: 98px;
        }
        .detail-card span {
            display: block;
            color: #969fab;
            font-size: .68rem;
            font-weight: 900;
            text-transform: uppercase;
        }
        .detail-card strong {
            display: block;
            margin-top: 7px;
            color: #ffffff;
            font-size: .94rem;
            overflow-wrap: anywhere;
        }
        .spec-band {
            border: 1px solid var(--line);
            border-radius: 7px;
            background: #101319;
            padding: 14px;
            margin-bottom: 12px;
        }
        .spec-band h3 {
            margin: 0 0 10px;
            color: #ffffff;
            font-size: 1rem;
        }
        .reward-row {
            display: grid;
            grid-template-columns: minmax(140px, 1fr) auto;
            gap: 12px;
            align-items: center;
            padding: 8px 2px;
            border-top: 1px solid #2e343d;
            color: #cdd3dc;
            font-size: .84rem;
        }
        .reward-row strong {
            color: var(--signal);
            font-family: "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
        }
        [data-testid="stAlert"] {
            border-radius: 6px;
            border: 1px solid #3d4652;
            background: #151a21;
            color: #eef2f7;
        }
        [data-testid="stDataFrame"],
        [data-testid="stVegaLiteChart"] {
            border: 1px solid var(--line);
            border-radius: 7px;
            overflow: hidden;
            background: #101319;
        }
        [data-baseweb="input"] > div,
        [data-baseweb="select"] > div {
            border-color: #46505c !important;
            background: #171b22 !important;
            color: #ffffff !important;
        }
        [data-testid="stNumberInput"] [data-baseweb="input"],
        [data-testid="stNumberInput"] [data-baseweb="base-input"] {
            background: #171b22 !important;
            border-color: #3b82f6 !important;
        }
        [data-testid="stNumberInput"] input {
            background: #171b22 !important;
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
            caret-color: #38bdf8 !important;
            opacity: 1 !important;
            font-weight: 800 !important;
        }
        [data-testid="stNumberInput"] button {
            background: #202630 !important;
            border-color: #46505c !important;
            color: #f8fafc !important;
        }
        [data-testid="stNumberInput"] button:hover {
            background: #293241 !important;
            border-color: #38bdf8 !important;
            color: #ffffff !important;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 7px;
            background: #11151b;
        }
        @media (min-width: 761px) and (max-width: 1100px) {
            .room-select-card {
                height: 570px;
                min-height: 570px;
            }
        }
        @media (max-width: 760px) {
            .room-select-card {
                height: auto;
                min-height: 0;
            }
            .topbar-inner {
                align-items: flex-start;
            }
            .topbar-badges {
                display: none;
            }
            .topbar {
                position: static;
            }
            .select-head {
                grid-template-columns: 1fr;
                align-items: start;
            }
            .campaign-score {
                text-align: left;
            }
            .hud, .objective-row {
                grid-template-columns: 1fr;
            }
            .param-grid {
                grid-template-columns: 1fr;
            }
            .arcade-shell {
                padding: 10px;
            }
            .maze {
                gap: 3px;
            }
            .section-head {
                display: block;
            }
            .section-head p {
                margin-top: 7px;
                text-align: left;
            }
            .details-grid {
                grid-template-columns: 1fr 1fr;
            }
            .game-card {
                min-height: 300px;
                padding: 18px;
            }
            .game-card::before {
                background:
                    linear-gradient(180deg, rgba(5,7,10,.90) 0%, rgba(5,7,10,.76) 62%, rgba(5,7,10,.56) 100%),
                    linear-gradient(90deg, rgba(5,7,10,.58), rgba(5,7,10,.16));
            }
            .game-card h2 {
                font-size: 1.65rem;
            }
            div[data-testid="stTabs"] [role="tablist"] {
                overflow-x: auto;
            }
            div[data-testid="stTabs"] button[role="tab"] {
                min-width: 104px;
            }
            .st-key-room_navigation [data-testid="stHorizontalBlock"] {
                flex-direction: row !important;
                flex-wrap: wrap !important;
                gap: 8px !important;
            }
            .st-key-room_navigation [data-testid="stColumn"] {
                width: auto !important;
                min-width: 0 !important;
                flex: 1 1 calc(33.333% - 8px) !important;
            }
            .st-key-room_navigation [data-testid="stColumn"]:nth-child(3) {
                order: -1;
                flex: 1 0 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_vars(room_kind: str) -> str:
    t = ROOM_THEMES[room_kind]
    return "; ".join(
        [
            f"--accent:{t['accent']}",
            f"--wall:{t['wall']}",
            f"--floor:{t['floor']}",
            f"--path:{t['path']}",
            f"--danger:{t['danger']}",
            f"--key:{t['key']}",
            f"--portal:{t['portal']}",
            f"--agent:{t['agent_color']}",
        ]
    )


def thumbnail_markup(room_kind: str) -> str:
    thumbnails = {
        "dp": '<span class="thumb-name">ICE MAZE</span>',
        "sarsa": '<span class="thumb-name">VAULT PUZZLE</span>',
        "q_learning": '<span class="thumb-name">REACTOR BLAST</span>',
        "approx": '<span class="thumb-name">LUNAR LANDING</span>',
        "obstacles": '<span class="thumb-name">PORTAL RUN</span>',
    }
    return thumbnails[room_kind]


def metric_dataframe(metrics: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(metrics)
    if "reward" in df:
        df["reward_ma_25"] = moving_average(df["reward"].astype(float).tolist(), 25)
    if "success" in df:
        df["success_rate_ma_25"] = moving_average(df["success"].astype(float).tolist(), 25)
    return df


def run_success_rate(run: Dict[str, Any]) -> float:
    attempts = run["result"].get("attempts", [])
    if not attempts:
        return 0.0
    if run["room_kind"] == "dp":
        sample = attempts
    else:
        sample = attempts[-min(50, len(attempts)) :]
    return sum(bool(attempt["success"]) for attempt in sample) / len(sample)


def mark_room_completed(room_kind: str) -> None:
    completed = st.session_state.setdefault("completed_rooms", [])
    if room_kind not in completed:
        completed.append(room_kind)


def next_room_kind(room_kind: str) -> str | None:
    index = ROOM_ORDER.index(room_kind)
    return ROOM_ORDER[index + 1] if index + 1 < len(ROOM_ORDER) else None


def header(room_kind: str | None) -> None:
    completed = len(set(st.session_state.get("completed_rooms", [])))
    current_label = ROOM_THEMES[room_kind]["title"] if room_kind else "Room Select"
    stage_label = "Mission active" if room_kind else "Campaign map"
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-inner">
            <div class="brand-lockup">
              <span class="brand-mark">RL</span>
              <div>
                <h1>RL Escape Lab</h1>
                <p>Agent Training Arcade</p>
              </div>
            </div>
            <div class="topbar-badges">
              <span class="topbar-badge active">Current<strong>{html.escape(current_label)}</strong></span>
              <span class="topbar-badge">Campaign<strong>{completed} / {len(ROOM_ORDER)} cleared</strong></span>
              <span class="topbar-badge">Session<strong><i class="status-live"></i>{stage_label}</strong></span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def room_intro(room_kind: str) -> None:
    t = ROOM_THEMES[room_kind]
    room_number = ROOM_ORDER.index(room_kind) + 1
    model_label = "Known model" if room_kind == "dp" else "Unknown model"
    representation = "10x10 grid" if room_kind in {"dp", "sarsa", "q_learning"} else "Continuous 10x10m"
    objectives = "".join(f'<div class="objective">{html.escape(item)}</div>' for item in t["objectives"])
    st.markdown(
        f"""
        <div class="game-card" style="{style_vars(room_kind)};background-image:url('app/static/game_art/{html.escape(t['banner_art'])}');background-position:right center;">
          <div class="mission-kicker">
            <span class="room-index">Room {room_number:02d}</span>
            <span>{model_label}</span>
            <span>{representation}</span>
          </div>
          <h2>{html.escape(t["title"])}</h2>
          <div class="sub">{html.escape(t["subtitle"])}</div>
          <div class="mission">{html.escape(t["mission"])}</div>
          <div class="objective-row">{objectives}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hud(room_kind: str, values: List[Tuple[str, str]], highlight: int = 0) -> None:
    cards = []
    for index, (label, value) in enumerate(values):
        klass = "hud-card highlight" if index == highlight else "hud-card"
        cards.append(
            f'<div class="{klass}"><div class="hud-label">{html.escape(label)}</div><div class="hud-value">{html.escape(value)}</div></div>'
        )
    st.markdown(f'<div class="hud" style="{style_vars(room_kind)}">{"".join(cards)}</div>', unsafe_allow_html=True)


def section_header(kicker: str, title: str, description: str = "") -> None:
    description_markup = f"<p>{html.escape(description)}</p>" if description else ""
    st.markdown(
        f"""
        <div class="section-head">
          <div>
            <div class="section-kicker">{html.escape(kicker)}</div>
            <h2>{html.escape(title)}</h2>
          </div>
          {description_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(symbol: str, title: str, message: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
          <div class="empty-icon">{html.escape(symbol)}</div>
          <div><strong>{html.escape(title)}</strong><span>{html.escape(message)}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def grid_label(env: GridEscapeRoom, pos: Tuple[int, int], state: GridState, room_kind: str) -> Tuple[str, str]:
    t = ROOM_THEMES[room_kind]
    labels = t.get("labels", {})
    if pos in env.config.walls:
        return "", "wall"
    for idx, key_pos in enumerate(env.config.keys):
        if pos == key_pos and not (state[2] & (1 << idx)):
            if room_kind == "q_learning":
                return f"C{idx + 1}", "key"
            return labels.get("key", "KEY"), "key"
    if pos in env.config.portals:
        return labels.get("portal", "WARP"), "portal"
    if pos in env.config.traps:
        return "", "trap"
    if pos in env.config.bonuses:
        return labels.get("bonus", "BON"), "bonus"
    if pos in env.config.slippery:
        return labels.get("ice", "ICE"), "slippery"
    if pos == env.config.goal:
        return labels.get("goal", "GOAL"), "goal"
    if pos == env.config.start:
        return labels.get("start", "START"), "start"
    return "", "empty"


def render_grid_game(env: GridEscapeRoom, trajectory: List[Dict[str, object]], step_index: int, room_kind: str) -> None:
    t = ROOM_THEMES[room_kind]
    current = trajectory[min(step_index, len(trajectory) - 1)]["state"]
    current_pos = (current[0], current[1])
    path = {(item["state"][0], item["state"][1]) for item in trajectory[: step_index + 1]}
    guards = set(env.guard_positions(current[3]))
    cells = []
    for row in range(env.rows):
        for col in range(env.cols):
            pos = (row, col)
            label, klass = grid_label(env, pos, current, room_kind)
            if pos == current_pos:
                klass = "agent"
                content = '<div class="pac"></div>'
            elif pos in guards:
                klass = "guard"
                content = f'<span class="label">{html.escape(t.get("labels", {}).get("guard", "SEC"))}</span>'
            elif label:
                content = f'<span class="label">{html.escape(label)}</span>'
            else:
                content = ""
            visited = " visited" if pos in path and klass == "empty" else ""
            cells.append(f'<div class="tile {klass}{visited}">{content}</div>')

    st.markdown(
        f"""
        <div class="arcade-shell" style="{style_vars(room_kind)}">
          <div class="arcade-title"><span>{html.escape(t["title"])}</span><span>{html.escape(t["algorithm"])}</span></div>
          <div class="maze">{"".join(cells)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_arena(env: ContinuousEscapeRoom, trajectory: List[Dict[str, object]], step_index: int, room_kind: str) -> None:
    t = ROOM_THEMES[room_kind]
    size = env.config.room_size
    current_index = min(step_index, len(trajectory) - 1)
    states = [np.asarray(item["state"], dtype=float) for item in trajectory[: current_index + 1]]
    current = states[-1]
    dots = []
    stride = max(1, len(states) // 140)
    for state in states[::stride]:
        left = state[0] / size * 100
        bottom = state[1] / size * 100
        dots.append(f'<div class="arena-path" style="left:{left:.2f}%; bottom:{bottom:.2f}%"></div>')

    hazards = []
    for x1, y1, x2, y2 in env.config.hazards:
        hazards.append(
            f'<div class="arena-hazard" style="left:{x1 / size * 100:.2f}%; bottom:{y1 / size * 100:.2f}%; width:{(x2 - x1) / size * 100:.2f}%; height:{(y2 - y1) / size * 100:.2f}%"></div>'
        )

    obstacles = []
    if isinstance(env, DynamicObstacleRoom):
        half = env.config.obstacle_width / 2
        step_obstacles = trajectory[current_index].get("obstacles") or env.obstacles
        for obstacle in step_obstacles:
            obstacles.append(
                f'<div class="arena-obstacle" style="left:{(obstacle["x"] - half) / size * 100:.2f}%; bottom:{(obstacle["y"] - half) / size * 100:.2f}%; width:{env.config.obstacle_width / size * 100:.2f}%; height:{env.config.obstacle_width / size * 100:.2f}%"></div>'
            )

    goal_x, goal_y = env.config.goal
    radius = env.config.goal_radius / size * 200
    agent_x = current[0] / size * 100
    agent_y = current[1] / size * 100
    st.markdown(
        f"""
        <div class="arena" style="{style_vars(room_kind)}">
          {"".join(hazards)}
          {"".join(obstacles)}
          {"".join(dots)}
          <div class="arena-goal" style="left:{goal_x / size * 100:.2f}%; bottom:{goal_y / size * 100:.2f}%; width:{radius:.2f}%; height:{radius:.2f}%"></div>
          <div class="arena-agent" style="left:{agent_x:.2f}%; bottom:{agent_y:.2f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def trajectory_score(trajectory: List[Dict[str, object]], step_index: int) -> float:
    return sum(float(item["reward"]) for item in trajectory[: step_index + 1])


def grid_hud(env: GridEscapeRoom, trajectory: List[Dict[str, object]], step_index: int, room_kind: str) -> None:
    current = trajectory[min(step_index, len(trajectory) - 1)]["state"]
    collected = int(current[2]).bit_count()
    required = len(env.config.keys)
    done = bool(trajectory[min(step_index, len(trajectory) - 1)].get("done"))
    hud(
        room_kind,
        [
            ("score", f"{trajectory_score(trajectory, step_index):.0f}"),
            ("step", str(step_index)),
            ("items", "none" if required == 0 else f"{collected}/{required}"),
            ("state", f"{current[0]},{current[1]}"),
            ("status", "won" if done else "playing"),
        ],
    )


def arena_hud(env: ContinuousEscapeRoom, trajectory: List[Dict[str, object]], step_index: int, room_kind: str) -> None:
    current = np.asarray(trajectory[min(step_index, len(trajectory) - 1)]["state"], dtype=float)
    dist = float(np.hypot(env.config.goal[0] - current[0], env.config.goal[1] - current[1]))
    visible = "n/a"
    if isinstance(env, DynamicObstacleRoom):
        visible = "yes" if len(current) > 6 and current[6] > 0.5 else "no"
    hud(
        room_kind,
        [
            ("score", f"{trajectory_score(trajectory, step_index):.1f}"),
            ("step", str(step_index)),
            ("distance", f"{dist:.2f}m"),
            ("velocity", f"{int(current[2])},{int(current[3])}"),
            ("sensor", visible),
        ],
    )


def make_env(room_kind: str, seed: int = 7, slip: float = 0.2, obstacle_count: int = 7, observation_range: float = 3.0):
    if room_kind == "dp":
        return GridEscapeRoom(room1_config(slip_probability=slip, seed=seed))
    if room_kind == "sarsa":
        return SokobanEscapeRoom(room2_config(slip_probability=slip, seed=seed))
    if room_kind == "q_learning":
        return GridEscapeRoom(room3_config(slip_probability=slip, seed=seed))
    if room_kind == "approx":
        return ContinuousEscapeRoom(continuous_room_config(seed=seed))
    return DynamicObstacleRoom(obstacle_room_config(seed=seed, obstacle_count=obstacle_count, observation_range=observation_range))


def arcade_payload(room_kind: str, replay_attempt: Dict[str, Any] | None = None) -> Dict[str, Any]:
    env = make_env(room_kind, seed=43, slip=0.2, obstacle_count=8, observation_range=3.4)
    theme = ROOM_THEMES[room_kind]
    base: Dict[str, Any] = {
        "kind": room_kind,
        "title": theme["title"],
        "subtitle": theme["subtitle"],
        "algorithm": theme["algorithm"],
        "mission": theme["mission"],
        "objectives": theme["objectives"],
        "artUrl": f"app/static/game_art/{theme['art']}",
        "agent": theme["agent"],
        "goalLabel": theme["goal"],
        "colors": {
            "accent": theme["accent"],
            "agent": theme["agent_color"],
            "wall": theme["wall"],
            "floor": theme["floor"],
            "path": theme["path"],
            "danger": theme["danger"],
            "key": theme["key"],
            "portal": theme["portal"],
        },
    }
    if isinstance(env, GridEscapeRoom):
        base.update(
            {
                "mode": "grid",
                "rows": env.rows,
                "cols": env.cols,
                "start": list(env.config.start),
                "goal": list(env.config.goal),
                "walls": [list(pos) for pos in sorted(env.config.walls)],
                "slippery": [list(pos) for pos in sorted(env.config.slippery)],
                "traps": [list(pos) for pos in sorted(env.config.traps)],
                "bonuses": [list(pos) for pos in sorted(env.config.bonuses)],
                "keys": [list(pos) for pos in env.config.keys],
                "portals": [{"from": list(k), "to": list(v)} for k, v in env.config.portals.items()],
                "guardCycles": [[list(pos) for pos in cycle] for cycle in env.config.guard_cycles],
                "boxStart": list(env.config.box_start) if env.config.box_start is not None else None,
                "boxTarget": list(env.config.box_target) if env.config.box_target is not None else None,
                "slipProbability": env.config.slip_probability,
                "stepReward": env.config.step_reward,
                "goalReward": env.config.goal_reward,
                "keyReward": env.config.key_reward,
                "blockedGoalPenalty": env.config.blocked_goal_penalty,
                "guardReward": env.config.guard_reward,
                "portalReward": env.config.portal_reward,
                "trapRewards": {f"{pos[0]},{pos[1]}": reward for pos, reward in env.config.traps.items()},
                "labels": theme.get("labels", {}),
            }
        )
    else:
        base.update(
            {
                "mode": "continuous",
                "roomSize": env.config.room_size,
                "start": list(env.config.start),
                "goal": list(env.config.goal),
                "goalRadius": env.config.goal_radius,
                "hazards": [list(item) for item in env.config.hazards],
                "obstacleMode": isinstance(env, DynamicObstacleRoom),
                "teleportMode": False,
                "obstacleCount": getattr(env.config, "obstacle_count", 0),
                "teleportCount": max(4, getattr(env.config, "obstacle_count", 7)),
                "obstacleWidth": getattr(env.config, "obstacle_width", 0.5),
                "observationRange": getattr(env.config, "observation_range", 3.0),
                "stepReward": env.config.step_reward,
                "goalReward": env.config.goal_reward,
                "wallPenalty": env.config.wall_penalty,
                "hazardPenalty": env.config.hazard_penalty,
                "progressScale": env.config.progress_scale,
                "obstaclePenalty": getattr(env.config, "obstacle_penalty", 0.0),
            }
        )
    if replay_attempt is not None:
        rewards = np.asarray(replay_attempt["rewards"], dtype=float)
        base["replay"] = {
            "episode": int(replay_attempt["episode"]),
            "states": np.asarray(replay_attempt["states"]).tolist(),
            "actions": np.asarray(replay_attempt["actions"], dtype=int).tolist(),
            "scores": np.cumsum(rewards).round(3).tolist(),
            "reward": float(replay_attempt["reward"]),
            "success": bool(replay_attempt["success"]),
            "epsilon": float(replay_attempt.get("epsilon", 0.0)),
        }
        if "obstacles" in replay_attempt:
            base["replay"]["obstacles"] = np.asarray(replay_attempt["obstacles"], dtype=float).tolist()
    return base


def arcade_component(room_kind: str, replay_attempt: Dict[str, Any] | None = None) -> None:
    config = json.dumps(arcade_payload(room_kind, replay_attempt), ensure_ascii=False)
    template = r"""
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <div class="real-game" tabindex="0">
      <style>
        .real-game {
          --accent: #38bdf8;
          --panel: #08111f;
          --text: #f8fafc;
          --muted: #aeb9c9;
          font-family: Inter, Segoe UI, Arial, sans-serif;
          color: var(--text);
          background: linear-gradient(180deg, #040814 0%, #08111f 100%);
          border: 1px solid #263244;
          border-radius: 18px;
          padding: 14px;
          overflow: hidden;
          position: relative;
          box-shadow: 0 24px 70px rgba(0, 0, 0, .38), inset 0 0 0 1px rgba(255,255,255,.025);
          transition: border-color .18s ease, box-shadow .18s ease;
        }
        .real-game.keyboard-active {
          border-color: color-mix(in srgb, var(--accent), white 18%);
          box-shadow: 0 24px 70px rgba(0, 0, 0, .42), 0 0 32px color-mix(in srgb, var(--accent), transparent 72%);
        }
        .game-grid {
          display: grid;
          grid-template-columns: minmax(480px, 1fr) 310px;
          gap: 14px;
          align-items: stretch;
        }
        .canvas-wrap {
          position: relative;
          width: 100%;
          max-width: 820px;
          align-self: start;
          overflow: hidden;
          border-radius: 12px;
        }
        .canvas-wrap::after {
          content: "";
          position: absolute;
          inset: 2px;
          z-index: 4;
          pointer-events: none;
          border-radius: 10px;
          background: repeating-linear-gradient(180deg, transparent 0 3px, rgba(255,255,255,.018) 3px 4px);
          mix-blend-mode: screen;
        }
        canvas {
          display: block;
          width: 100%;
          aspect-ratio: 4 / 3;
          border-radius: 12px;
          border: 2px solid var(--accent);
          background: #020617;
          box-shadow: 0 0 28px color-mix(in srgb, var(--accent), transparent 55%);
          outline: none;
        }
        .screen-status {
          position: absolute;
          z-index: 7;
          right: 12px;
          bottom: 12px;
          display: flex;
          gap: 7px;
          align-items: center;
          pointer-events: none;
        }
        .screen-chip {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          min-height: 28px;
          padding: 0 9px;
          border: 1px solid rgba(148,163,184,.35);
          border-radius: 6px;
          background: rgba(2,6,23,.78);
          color: #cbd5e1;
          font-size: .68rem;
          font-weight: 900;
          backdrop-filter: blur(8px);
        }
        .screen-chip .signal {
          width: 7px;
          height: 7px;
          border-radius: 50%;
          background: #64748b;
          box-shadow: 0 0 0 3px rgba(100,116,139,.13);
        }
        .keyboard-active .screen-chip .signal {
          background: #22c55e;
          box-shadow: 0 0 12px rgba(34,197,94,.9);
          animation: signalPulse 1.25s ease-in-out infinite;
        }
        .screen-flash {
          position: absolute;
          inset: 2px;
          z-index: 8;
          border-radius: 10px;
          pointer-events: none;
          opacity: 0;
        }
        .canvas-wrap.hit canvas { animation: gameShake .26s ease-in-out; }
        .canvas-wrap.hit .screen-flash {
          background: rgba(239,68,68,.24);
          animation: flashFrame .34s ease-out;
        }
        .canvas-wrap.reward .screen-flash {
          background: rgba(45,212,191,.18);
          animation: flashFrame .34s ease-out;
        }
        .canvas-wrap.win .screen-flash {
          background: rgba(34,197,94,.22);
          animation: flashFrame .7s ease-out;
        }
        @keyframes gameShake {
          0%, 100% { transform: translate(0, 0); }
          22% { transform: translate(-6px, 2px); }
          48% { transform: translate(5px, -2px); }
          72% { transform: translate(-3px, 1px); }
        }
        @keyframes flashFrame {
          0% { opacity: 0; }
          20% { opacity: 1; }
          100% { opacity: 0; }
        }
        @keyframes signalPulse {
          50% { transform: scale(.72); opacity: .72; }
        }
        .game-overlay {
          position: absolute;
          inset: 2px;
          z-index: 10;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 22px;
          border-radius: 11px;
          background: rgba(2, 6, 23, .72);
          backdrop-filter: blur(7px);
        }
        .game-overlay.hidden { display: none; }
        .overlay-panel {
          width: min(88%, 430px);
          border: 1px solid color-mix(in srgb, var(--accent), white 18%);
          border-radius: 12px;
          padding: 22px;
          text-align: center;
          background: rgba(8, 17, 31, .94);
          box-shadow: 0 22px 70px rgba(0,0,0,.48), 0 0 24px color-mix(in srgb, var(--accent), transparent 76%);
        }
        .overlay-kicker {
          color: var(--accent);
          font-size: .74rem;
          font-weight: 900;
          text-transform: uppercase;
          margin-bottom: 7px;
        }
        .overlay-panel h3 {
          margin: 0;
          color: #fff;
          font-size: 1.65rem;
        }
        .overlay-panel p {
          margin: 9px 0 16px;
          color: #cbd5e1;
          line-height: 1.45;
        }
        .overlay-panel button {
          width: 100%;
          min-height: 46px;
          border: 1px solid var(--accent);
          border-radius: 8px;
          color: #fff;
          background: color-mix(in srgb, var(--accent), #0f172a 58%);
          font-weight: 900;
          cursor: pointer;
        }
        .side {
          border: 1px solid #263244;
          border-radius: 16px;
          background: rgba(15, 23, 42, .92);
          padding: 14px;
          min-height: 100%;
          box-shadow: inset 0 0 0 1px rgba(255,255,255,.025);
        }
        .side h2 {
          margin: 0;
          font-size: 1.25rem;
          color: #f8fafc;
        }
        .side .sub {
          color: var(--accent);
          font-weight: 900;
          margin: 4px 0 10px 0;
        }
        .side p {
          color: #cbd5e1;
          line-height: 1.45;
          margin: 0 0 12px 0;
          font-size: .93rem;
        }
        .stats {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          margin: 12px 0;
        }
        .stat {
          border: 1px solid #263244;
          border-radius: 10px;
          padding: 8px;
          background: rgba(255,255,255,.04);
        }
        .stat b {
          display: block;
          color: #94a3b8;
          font-size: .72rem;
          text-transform: uppercase;
        }
        .stat span {
          display: block;
          color: #f8fafc;
          font-size: 1.02rem;
          font-weight: 900;
          margin-top: 3px;
        }
        .msg {
          min-height: 54px;
          border: 1px solid color-mix(in srgb, var(--accent), #263244 50%);
          border-radius: 10px;
          background: rgba(255,255,255,.04);
          padding: 10px;
          color: #e5e7eb;
          line-height: 1.35;
          margin-bottom: 12px;
        }
        .mission-meter {
          border: 1px solid #263244;
          border-radius: 10px;
          background: rgba(2,6,23,.42);
          padding: 10px;
          margin: 10px 0 12px;
        }
        .mission-meter-head {
          display: flex;
          justify-content: space-between;
          align-items: center;
          color: #94a3b8;
          font-size: .7rem;
          font-weight: 900;
          text-transform: uppercase;
        }
        .mission-meter-head strong {
          color: var(--accent);
          font-size: .8rem;
        }
        .mission-track {
          height: 7px;
          overflow: hidden;
          margin: 8px 0 9px;
          border-radius: 4px;
          background: #1e293b;
        }
        .mission-track span {
          display: block;
          width: 0;
          height: 100%;
          border-radius: inherit;
          background: var(--accent);
          box-shadow: 0 0 14px color-mix(in srgb, var(--accent), transparent 30%);
          transition: width .24s ease;
        }
        .objective-list {
          display: grid;
          gap: 5px;
        }
        .objective-item {
          display: grid;
          grid-template-columns: 15px minmax(0, 1fr);
          gap: 6px;
          align-items: center;
          color: #94a3b8;
          font-size: .76rem;
          font-weight: 800;
        }
        .objective-item i {
          width: 8px;
          height: 8px;
          border: 1px solid #64748b;
          border-radius: 50%;
          background: transparent;
        }
        .objective-item.active { color: #e2e8f0; }
        .objective-item.active i {
          border-color: var(--accent);
          box-shadow: 0 0 8px color-mix(in srgb, var(--accent), transparent 35%);
        }
        .objective-item.done { color: #bbf7d0; }
        .objective-item.done i {
          border-color: #22c55e;
          background: #22c55e;
          box-shadow: 0 0 8px rgba(34,197,94,.68);
        }
        .keyboard-console {
          border: 1px solid #334155;
          border-radius: 10px;
          background: rgba(2,6,23,.48);
          padding: 9px;
          margin-top: 10px;
        }
        .keyboard-console-head {
          display: flex;
          align-items: center;
          justify-content: space-between;
          color: #94a3b8;
          font-size: .68rem;
          font-weight: 900;
          text-transform: uppercase;
        }
        .keyboard-console-head span:last-child { color: var(--accent); }
        .key-cluster {
          display: grid;
          grid-template-columns: repeat(3, 34px);
          justify-content: center;
          gap: 5px;
          margin-top: 8px;
        }
        .key-cluster kbd {
          display: grid;
          place-items: center;
          width: 34px;
          height: 30px;
          border: 1px solid #475569;
          border-bottom-width: 3px;
          border-radius: 6px;
          background: #111827;
          color: #f8fafc;
          font-family: inherit;
          font-size: .88rem;
          font-weight: 900;
          transition: transform .08s ease, border-color .08s ease, background .08s ease;
        }
        .key-cluster kbd.blank { visibility: hidden; }
        .key-cluster kbd.active {
          transform: translateY(2px);
          border-color: var(--accent);
          border-bottom-width: 1px;
          background: color-mix(in srgb, var(--accent), #0f172a 72%);
          box-shadow: 0 0 13px color-mix(in srgb, var(--accent), transparent 55%);
        }
        .controls {
          display: none;
          grid-template-columns: repeat(3, 1fr);
          gap: 8px;
          margin-top: 10px;
        }
        .controls button, .reset {
          min-height: 42px;
          border: 1px solid var(--accent);
          border-radius: 10px;
          background: linear-gradient(135deg, #1d4ed8, #111827);
          color: #ffffff;
          font-weight: 900;
          cursor: pointer;
          box-shadow: 0 8px 20px rgba(0,0,0,.25);
        }
        .controls button:hover, .reset:hover {
          border-color: #facc15;
          background: linear-gradient(135deg, #2563eb, #172554);
        }
        .reset {
          width: 100%;
          margin-top: 0;
        }
        .action-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
          margin-top: 8px;
        }
        .wide {
          grid-column: span 3;
        }
        .hidden {
          display: none;
        }
        .help {
          margin-top: 11px;
          color: #94a3b8;
          font-size: .84rem;
          line-height: 1.35;
        }
        .replay-controls {
          display: grid;
          grid-template-columns: 44px 1fr 44px;
          gap: 8px;
          align-items: center;
          margin-top: 10px;
        }
        .replay-controls button {
          min-height: 42px;
          border: 1px solid var(--accent);
          border-radius: 9px;
          background: #111827;
          color: #fff;
          font-weight: 900;
          cursor: pointer;
        }
        .replay-controls input[type="range"] {
          width: 100%;
          accent-color: var(--accent);
        }
        .replay-speed {
          display: grid;
          grid-template-columns: repeat(4, 1fr);
          gap: 6px;
          margin-top: 8px;
        }
        .replay-speed button {
          border: 1px solid #334155;
          border-radius: 7px;
          background: #0f172a;
          color: #cbd5e1;
          min-height: 34px;
          font-weight: 800;
          cursor: pointer;
        }
        .replay-speed button.active {
          border-color: var(--accent);
          color: #fff;
          background: color-mix(in srgb, var(--accent), #0f172a 76%);
        }
        @media (max-width: 900px) {
          .game-grid {
            grid-template-columns: 1fr;
            gap: 8px;
          }
          .side {
            padding: 10px;
          }
          .side h2,
          .side .sub,
          .side > #gameMission,
          .help {
            display: none;
          }
          .stats {
            margin: 0 0 8px;
          }
          .msg {
            min-height: 44px;
            padding: 8px 10px;
            margin-bottom: 8px;
          }
          .mission-meter {
            padding: 8px 10px;
            margin: 8px 0;
          }
          .controls:not(.hidden) {
            display: grid;
            margin-top: 8px;
          }
          .keyboard-console {
            display: none;
          }
        }
      </style>
      <div class="game-grid">
        <div class="canvas-wrap">
          <canvas id="gameCanvas" width="820" height="615" tabindex="0"></canvas>
          <div class="screen-status">
            <span class="screen-chip"><span class="signal"></span><span id="keyboardState">KEYBOARD STANDBY</span></span>
            <span class="screen-chip" id="runState">LIVE RUN</span>
          </div>
          <div class="screen-flash"></div>
          <div class="game-overlay" id="gameOverlay">
            <div class="overlay-panel">
              <div class="overlay-kicker" id="overlayKicker">ARCADE MISSION</div>
              <h3 id="overlayTitle">Ready?</h3>
              <p id="overlayCopy">Enter the arena and complete the objective.</p>
              <button id="overlayAction">START MISSION</button>
            </div>
          </div>
        </div>
        <div class="side">
          <h2 id="gameTitle"></h2>
          <div class="sub" id="gameSub"></div>
          <p id="gameMission"></p>
          <div class="stats">
            <div class="stat"><b>Score</b><span id="score">0</span></div>
            <div class="stat"><b>Steps</b><span id="steps">0</span></div>
            <div class="stat"><b>Goal</b><span id="goal">-</span></div>
            <div class="stat"><b>Status</b><span id="status">Ready</span></div>
          </div>
          <div class="msg" id="message">Click the game and use keyboard controls.</div>
          <div class="mission-meter">
            <div class="mission-meter-head"><span>Mission progress</span><strong id="missionPercent">0%</strong></div>
            <div class="mission-track"><span id="missionFill"></span></div>
            <div class="objective-list" id="objectiveList"></div>
          </div>
          <div class="keyboard-console" id="keyboardConsole">
            <div class="keyboard-console-head"><span>Keyboard control</span><span>ARROWS / WASD</span></div>
            <div class="key-cluster">
              <kbd class="blank"></kbd><kbd data-keycap="up">&#8593;</kbd><kbd class="blank"></kbd>
              <kbd data-keycap="left">&#8592;</kbd><kbd data-keycap="down">&#8595;</kbd><kbd data-keycap="right">&#8594;</kbd>
            </div>
          </div>
          <div id="gridControls" class="controls">
            <div></div><button data-grid="0">UP</button><div></div>
            <button data-grid="3">LEFT</button><button data-grid="2">DOWN</button><button data-grid="1">RIGHT</button>
          </div>
          <div id="vectorControls" class="controls hidden">
            <button data-v="-1,1">UP LEFT</button><button data-v="0,1">UP</button><button data-v="1,1">UP RIGHT</button>
            <button data-v="-1,0">LEFT</button><button data-v="0,0">STOP</button><button data-v="1,0">RIGHT</button>
            <button data-v="-1,-1">DOWN LEFT</button><button data-v="0,-1">DOWN</button><button data-v="1,-1">DOWN RIGHT</button>
          </div>
          <div id="replayPanel" class="hidden">
            <div class="replay-controls">
              <button id="replayPrev" title="Previous step">&#9664;</button>
              <button id="replayToggle">PAUSE</button>
              <button id="replayNext" title="Next step">&#9654;</button>
              <div></div><input id="replayTimeline" type="range" min="0" value="0"><div></div>
            </div>
            <div class="replay-speed">
              <button data-speed="900">0.5x</button>
              <button data-speed="450" class="active">1x</button>
              <button data-speed="220">2x</button>
              <button data-speed="90">4x</button>
            </div>
          </div>
          <div class="action-row">
            <button class="reset" id="pauseBtn">PAUSE</button>
            <button class="reset" id="resetBtn">RESET</button>
          </div>
          <div class="help" id="gameHelp">Keyboard: arrows or WASD. In continuous rooms, hold keys to fly. In grid rooms, each key press moves one tile.</div>
        </div>
      </div>
      <script>
      const cfg = __CONFIG__;
      const root = document.currentScript.closest('.real-game');
      root.style.setProperty('--accent', cfg.colors.accent);
      root.dataset.kind = cfg.kind;
      const canvas = root.querySelector('#gameCanvas');
      const ctx = canvas.getContext('2d');
      const canvasWrap = root.querySelector('.canvas-wrap');
      const keyboardStateEl = root.querySelector('#keyboardState');
      const runStateEl = root.querySelector('#runState');
      const missionFillEl = root.querySelector('#missionFill');
      const missionPercentEl = root.querySelector('#missionPercent');
      const objectiveListEl = root.querySelector('#objectiveList');
      objectiveListEl.innerHTML = (cfg.objectives || []).map((item, index) =>
        '<div class="objective-item" data-objective="' + index + '"><i></i><span>' + item + '</span></div>'
      ).join('');
      const objectiveItems = Array.from(objectiveListEl.querySelectorAll('.objective-item'));
      function setKeyboardActive(active) {
        root.classList.toggle('keyboard-active', active);
        keyboardStateEl.textContent = active ? 'KEYBOARD ACTIVE' : 'KEYBOARD STANDBY';
      }
      function setKeycap(direction, active) {
        const keycap = root.querySelector('[data-keycap="' + direction + '"]');
        if (keycap) keycap.classList.toggle('active', active);
      }
      function screenEffect(kind) {
        canvasWrap.classList.remove('hit', 'reward', 'win');
        void canvasWrap.offsetWidth;
        canvasWrap.classList.add(kind);
        window.setTimeout(() => canvasWrap.classList.remove(kind), kind === 'win' ? 720 : 380);
      }
      function focusGame() {
        root.focus();
        setKeyboardActive(true);
      }
      const arenaArt = new Image();
      arenaArt.src = cfg.artUrl;
      arenaArt.onload = () => draw();
      const scoreEl = root.querySelector('#score');
      const stepsEl = root.querySelector('#steps');
      const goalEl = root.querySelector('#goal');
      const statusEl = root.querySelector('#status');
      const msgEl = root.querySelector('#message');
      root.querySelector('#gameTitle').textContent = cfg.title;
      root.querySelector('#gameSub').textContent = cfg.algorithm + ' | ' + cfg.subtitle;
      root.querySelector('#gameMission').textContent = cfg.mission;
      goalEl.textContent = cfg.goalLabel;
      const gameOverlay = root.querySelector('#gameOverlay');
      const overlayKicker = root.querySelector('#overlayKicker');
      const overlayTitle = root.querySelector('#overlayTitle');
      const overlayCopy = root.querySelector('#overlayCopy');
      const overlayAction = root.querySelector('#overlayAction');
      const pauseBtn = root.querySelector('#pauseBtn');
      overlayTitle.textContent = cfg.title;
      overlayCopy.textContent = cfg.mission;
      let gameStarted = false;
      let gamePaused = false;
      let resultShown = false;
      function showGameOverlay(kicker, title, copy, action) {
        overlayKicker.textContent = kicker;
        overlayTitle.textContent = title;
        overlayCopy.textContent = copy;
        overlayAction.textContent = action;
        gameOverlay.classList.remove('hidden');
      }
      function hideGameOverlay() {
        gameOverlay.classList.add('hidden');
      }
      if (cfg.mode === 'continuous') {
        root.querySelector('#gridControls').classList.add('hidden');
        root.querySelector('#vectorControls').classList.remove('hidden');
      }
      if (cfg.replay) {
        root.querySelector('#gridControls').classList.add('hidden');
        root.querySelector('#vectorControls').classList.add('hidden');
        root.querySelector('#keyboardConsole').classList.add('hidden');
        root.querySelector('#replayPanel').classList.remove('hidden');
        root.querySelector('#resetBtn').textContent = 'RESTART REPLAY';
        runStateEl.textContent = 'EPISODE REPLAY';
        keyboardStateEl.textContent = 'PLAYER CONTROL';
        pauseBtn.classList.add('hidden');
        gameStarted = true;
        hideGameOverlay();
        root.querySelector('#gameHelp').textContent = 'Replay controls: pause, inspect one step at a time, drag the timeline, or change playback speed.';
        root.querySelector('#gameSub').textContent = cfg.algorithm + ' | Training episode ' + cfg.replay.episode;
      }

      const W = canvas.width;
      const H = canvas.height;
      const keyOf = p => p[0] + ',' + p[1];
      const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
      function drawArtBase(darken=.32) {
        if (arenaArt.complete && arenaArt.naturalWidth > 0) {
          ctx.drawImage(arenaArt, 0, 0, W, H);
        } else {
          ctx.fillStyle = '#020617';
          ctx.fillRect(0, 0, W, H);
        }
        ctx.fillStyle = 'rgba(2,6,23,' + darken + ')';
        ctx.fillRect(0, 0, W, H);
      }
      const particles = [];
      function emitBurst(x, y, color, count=16) {
        for (let i = 0; i < count; i++) {
          const angle = Math.random() * Math.PI * 2;
          const speed = 1.2 + Math.random() * 3.2;
          particles.push({
            x, y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            life: 1,
            decay: .018 + Math.random() * .025,
            size: 2 + Math.random() * 4,
            color
          });
        }
      }
      function drawParticles() {
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        for (let i = particles.length - 1; i >= 0; i--) {
          const particle = particles[i];
          particle.x += particle.vx;
          particle.y += particle.vy;
          particle.vx *= .97;
          particle.vy *= .97;
          particle.life -= particle.decay;
          if (particle.life <= 0) { particles.splice(i, 1); continue; }
          ctx.globalAlpha = particle.life;
          ctx.fillStyle = particle.color;
          ctx.beginPath();
          ctx.arc(particle.x, particle.y, particle.size * particle.life, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();
      }
      function rr(x, y, w, h, r, fill, stroke) {
        ctx.beginPath();
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r);
        ctx.lineTo(x + w, y + h - r);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
        ctx.lineTo(x + r, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r);
        ctx.lineTo(x, y + r);
        ctx.quadraticCurveTo(x, y, x + r, y);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
      }
      function missionProgressValue() {
        if (cfg.mode === 'grid') {
          if (!grid) return 0;
          if (grid.won) return 1;
          const startDistance = Math.max(1, Math.abs(cfg.goal[0] - cfg.start[0]) + Math.abs(cfg.goal[1] - cfg.start[1]));
          const currentDistance = Math.abs(cfg.goal[0] - grid.pos[0]) + Math.abs(cfg.goal[1] - grid.pos[1]);
          const routeProgress = clamp(1 - currentDistance / startDistance, 0, 1);
          if (cfg.kind === 'sarsa') {
            const box = grid.boxes && grid.boxes[0];
            const target = grid.targets && grid.targets[0];
            const boxDistance = box && target ? Math.abs(box[0] - target[0]) + Math.abs(box[1] - target[1]) : 10;
            const boxProgress = sokobanSolved() ? 1 : clamp(1 - boxDistance / 10, 0, .82);
            return clamp(.06 + boxProgress * .66 + routeProgress * .22, 0, .94);
          }
          if (cfg.kind === 'q_learning') {
            const itemProgress = grid.keys.length ? grid.collected.size / grid.keys.length : 1;
            return clamp(.06 + itemProgress * .58 + routeProgress * .28, 0, .94);
          }
          return clamp(.06 + routeProgress * .88, 0, .94);
        }
        if (!cont) return 0;
        if (cont.won) return 1;
        const startDistance = Math.max(.001, Math.hypot(cfg.goal[0] - cfg.start[0], cfg.goal[1] - cfg.start[1]));
        const currentDistance = Math.hypot(cfg.goal[0] - cont.x, cfg.goal[1] - cont.y);
        return clamp(.04 + (1 - currentDistance / startDistance) * .9, 0, .94);
      }
      function updateMissionHud(status) {
        const progress = missionProgressValue();
        const percent = Math.round(progress * 100);
        missionFillEl.style.width = percent + '%';
        missionPercentEl.textContent = percent + '%';
        const completed = progress >= 1 ? objectiveItems.length : Math.floor(progress * objectiveItems.length);
        objectiveItems.forEach((item, index) => {
          item.classList.toggle('done', index < completed);
          item.classList.toggle('active', index === completed && progress < 1);
        });
        runStateEl.textContent = cfg.replay ? 'EPISODE REPLAY' : status;
      }
      function updateStats(score, steps, status, message) {
        scoreEl.textContent = Math.round(score).toString();
        stepsEl.textContent = steps.toString();
        statusEl.textContent = status;
        msgEl.textContent = message;
        updateMissionHud(status);
      }

      let grid = null;
      function initGrid() {
        grid = {
          pos: cfg.start.slice(),
          dir: 1,
          score: 0,
          steps: 0,
          anim: 0,
          lastEnemyTick: 0,
          lastGhostTick: 0,
          ghostMoveStarted: 0,
          ghostTick: 0,
          ghosts: [],
          ghostVisualFrom: [],
          ghostVisualTo: [],
          enemyHitCooldown: 0,
          playerMoved: false,
          visualFrom: cfg.start.slice(),
          visualTo: cfg.start.slice(),
          moveStarted: 0,
          hits: 0,
          won: false,
          phase: 0,
          collected: new Set(),
          bonuses: new Set(),
          visited: new Set([keyOf(cfg.start)]),
          message: 'Collect what the room requires, then reach ' + cfg.goalLabel + '.'
        };
        grid.walls = new Set(cfg.walls.map(keyOf));
        grid.slippery = new Set(cfg.slippery.map(keyOf));
        grid.traps = new Set(cfg.traps.map(keyOf));
        grid.bonusesSet = new Set(cfg.bonuses.map(keyOf));
        grid.keys = cfg.keys || [];
        grid.portals = new Map((cfg.portals || []).map(p => [keyOf(p.from), p.to]));
        grid.targets = cfg.boxTarget ? [cfg.boxTarget] : (cfg.keys || []);
        grid.boxes = cfg.kind === 'sarsa' && cfg.boxStart ? [cfg.boxStart.slice()] : [];
        if (cfg.kind === 'dp' && cfg.guardCycles) {
          grid.ghosts = cfg.guardCycles.map(cycle => cycle[0].slice());
          grid.ghostVisualFrom = grid.ghosts.map(position => position.slice());
          grid.ghostVisualTo = grid.ghosts.map(position => position.slice());
        }
        grid.pellets = new Set();
        grid.pelletsEaten = 0;
        grid.requiredPellets = 0;
        if (cfg.kind === 'sarsa') {
          grid.message = 'Sokoban rule: push the BOX onto TARGET, then enter SAFE.';
        } else if (cfg.kind === 'dp') {
          grid.message = 'Reach EXIT quickly. Ghost positions are part of the known model.';
        }
        draw();
      }
      function animateGridMove(from, to) {
        if (!grid) return;
        grid.visualFrom = from.slice();
        grid.visualTo = to.slice();
        grid.moveStarted = performance.now();
      }
      function guardPositions() {
        if (!cfg.guardCycles) return [];
        if (cfg.kind === 'dp' && !cfg.replay && grid.ghosts.length) {
          return grid.ghosts;
        }
        return cfg.guardCycles.map(cycle => cycle[grid.phase % cycle.length]);
      }
      function insideGrid(row, col) {
        return row >= 0 && row < cfg.rows && col >= 0 && col < cfg.cols;
      }
      function boxIndexAt(row, col) {
        return (grid.boxes || []).findIndex(box => box[0] === row && box[1] === col);
      }
      function sokobanSolved() {
        if (!grid.targets || !grid.targets.length) return true;
        const targets = new Set(grid.targets.map(keyOf));
        return grid.boxes.every(box => targets.has(keyOf(box)));
      }
      function resetLiveGhosts() {
        if (cfg.kind !== 'dp' || !cfg.guardCycles) return;
        grid.ghosts = cfg.guardCycles.map(cycle => cycle[0].slice());
        grid.ghostVisualFrom = grid.ghosts.map(position => position.slice());
        grid.ghostVisualTo = grid.ghosts.map(position => position.slice());
        grid.ghostMoveStarted = performance.now();
      }
      function ghostTarget(index) {
        if (index === 0) return grid.pos.slice();
        const deltas = [[-1,0],[0,1],[1,0],[0,-1]];
        const direction = deltas[grid.dir] || [0,0];
        const lookAhead = [
          clamp(grid.pos[0] + direction[0] * 2, 0, cfg.rows - 1),
          clamp(grid.pos[1] + direction[1] * 2, 0, cfg.cols - 1)
        ];
        return grid.walls.has(keyOf(lookAhead)) ? grid.pos.slice() : lookAhead;
      }
      function nextGhostStep(start, target, occupied) {
        const startKey = keyOf(start);
        const targetKey = keyOf(target);
        if (startKey === targetKey) return start.slice();
        const queue = [start.slice()];
        const previous = new Map([[startKey, null]]);
        const deltas = [[-1,0],[0,1],[1,0],[0,-1]];
        let found = false;
        for (let cursor = 0; cursor < queue.length && !found; cursor++) {
          const current = queue[cursor];
          for (const delta of deltas) {
            const candidate = [current[0] + delta[0], current[1] + delta[1]];
            const candidateKey = keyOf(candidate);
            if (!insideGrid(candidate[0], candidate[1]) || grid.walls.has(candidateKey) || previous.has(candidateKey)) continue;
            if (occupied.has(candidateKey) && candidateKey !== targetKey) continue;
            previous.set(candidateKey, keyOf(current));
            queue.push(candidate);
            if (candidateKey === targetKey) { found = true; break; }
          }
        }
        if (!previous.has(targetKey)) {
          const legal = deltas
            .map(delta => [start[0] + delta[0], start[1] + delta[1]])
            .filter(candidate => insideGrid(candidate[0], candidate[1]) && !grid.walls.has(keyOf(candidate)) && !occupied.has(keyOf(candidate)))
            .sort((a, b) =>
              (Math.abs(a[0] - target[0]) + Math.abs(a[1] - target[1])) -
              (Math.abs(b[0] - target[0]) + Math.abs(b[1] - target[1]))
            );
          return legal.length ? legal[0] : start.slice();
        }
        let cursorKey = targetKey;
        while (previous.get(cursorKey) !== startKey && previous.get(cursorKey) !== null) {
          cursorKey = previous.get(cursorKey);
        }
        return cursorKey.split(',').map(Number);
      }
      function advanceLiveGhosts(now) {
        if (cfg.kind !== 'dp' || cfg.replay || !grid.ghosts.length) return;
        if (!grid.lastGhostTick) grid.lastGhostTick = now;
        if (now - grid.lastGhostTick < 460) return;
        grid.lastGhostTick = now;
        const from = grid.ghosts.map(position => position.slice());
        const occupied = new Set(grid.ghosts.map(keyOf));
        const next = [];
        for (let index = 0; index < grid.ghosts.length; index++) {
          const current = grid.ghosts[index];
          occupied.delete(keyOf(current));
          const candidate = nextGhostStep(current, ghostTarget(index), occupied);
          occupied.add(keyOf(candidate));
          next.push(candidate);
        }
        grid.ghostVisualFrom = from;
        grid.ghostVisualTo = next.map(position => position.slice());
        grid.ghostMoveStarted = now;
        grid.ghosts = next;
        grid.ghostTick += 1;
        grid.phase = grid.ghostTick;
        grid.score += cfg.stepReward * .25;
        enemyCollision();
      }
      function enemyCollision() {
        if (cfg.kind !== 'dp') return false;
        if (grid.anim < grid.enemyHitCooldown) return false;
        const hit = guardPositions().some(enemy => keyOf(enemy) === keyOf(grid.pos));
        if (hit) {
          const caughtAt = grid.pos.slice();
          const hitCenter = cellCenter(grid.pos);
          emitBurst(hitCenter.x, hitCenter.y, '#fb7185', 28);
          grid.score += cfg.guardReward;
          grid.hits += 1;
          grid.pos = cfg.start.slice();
          animateGridMove(caughtAt, grid.pos);
          resetLiveGhosts();
          grid.enemyHitCooldown = grid.anim + 2200;
          grid.message = 'A ghost caught PAC. Back to start.';
          screenEffect('hit');
        }
        return hit;
      }
      function updateGridWorld(now) {
        if (!grid || grid.won) return;
        grid.anim = now || 0;
        if (cfg.kind === 'dp' && !cfg.replay) {
          advanceLiveGhosts(grid.anim);
          root.dataset.ghostPhase = String(grid.phase);
          return;
        }
        if (grid.lastEnemyTick === 0) grid.lastEnemyTick = grid.anim;
        if (grid.anim - grid.lastEnemyTick < 360) return;
        grid.lastEnemyTick = grid.anim;
      }
      function moveSokoban(action) {
        focusGame();
        if (!grid || grid.won) return;
        const oldPos = grid.pos.slice();
        if (grid.slippery.has(keyOf(grid.pos)) && Math.random() < cfg.slipProbability) {
          action = Math.random() < 0.5 ? (action + 1) % 4 : (action + 3) % 4;
          grid.message = 'Oil slick changed your push direction.';
        }
        const deltas = [[-1,0],[0,1],[1,0],[0,-1]];
        const d = deltas[action];
        grid.dir = action;
        const nr = grid.pos[0] + d[0];
        const nc = grid.pos[1] + d[1];
        grid.steps += 1;
        grid.score += cfg.stepReward;
        if (!insideGrid(nr, nc) || grid.walls.has(nr + ',' + nc)) {
          grid.score += cfg.blockedGoalPenalty;
          grid.message = 'Wall blocked the move.';
          draw();
          return;
        }
        const boxIndex = boxIndexAt(nr, nc);
        if (boxIndex >= 0) {
          const br = nr + d[0];
          const bc = nc + d[1];
          if (!insideGrid(br, bc) || grid.walls.has(br + ',' + bc) || boxIndexAt(br, bc) >= 0) {
            grid.message = 'BOX is blocked. Try another angle.';
            draw();
            return;
          }
          grid.boxes[boxIndex] = [br, bc];
          grid.pos = [nr, nc];
          const pushedBox = cellCenter([br, bc]);
          emitBurst(pushedBox.x, pushedBox.y, '#fde68a', 12);
          grid.message = 'BOX pushed.';
        } else {
          grid.pos = [nr, nc];
        }
        const here = keyOf(grid.pos);
        grid.visited.add(here);
        if (grid.traps.has(here)) {
          grid.score += cfg.trapRewards[here] || 0;
          grid.hits += 1;
          grid.message = 'Laser tile hit. Keep the crate route clean.';
          screenEffect('hit');
        }
        if (grid.bonusesSet.has(here) && !grid.bonuses.has(here)) {
          grid.bonuses.add(here);
          grid.score += 18;
          grid.message = 'Coin collected.';
        }
        if (sokobanSolved() && !grid.collected.has(0)) {
          grid.collected.add(0);
          grid.score += cfg.keyReward;
          const targetCenter = cellCenter(grid.targets[0]);
          emitBurst(targetCenter.x, targetCenter.y, '#86efac', 30);
          grid.message = 'BOX locked on TARGET. SAFE is open.';
          screenEffect('reward');
        }
        if (keyOf(grid.pos) === keyOf(cfg.goal)) {
          if (sokobanSolved()) {
            grid.score += cfg.goalReward;
            grid.won = true;
            const safeCenter = cellCenter(cfg.goal);
            emitBurst(safeCenter.x, safeCenter.y, '#86efac', 46);
            grid.message = 'Sokoban vault cleared.';
            screenEffect('win');
          } else {
            grid.score += cfg.blockedGoalPenalty;
            grid.message = 'SAFE is locked. Push BOX onto TARGET first.';
          }
        }
        animateGridMove(oldPos, grid.pos);
        draw();
      }
      function moveGrid(action) {
        focusGame();
        if (!grid || grid.won) return;
        const oldPos = grid.pos.slice();
        if (cfg.kind === 'sarsa') {
          moveSokoban(action);
          return;
        }
        grid.playerMoved = true;
        if (grid.slippery.has(keyOf(grid.pos)) && Math.random() < cfg.slipProbability) {
          action = Math.random() < 0.5 ? (action + 1) % 4 : (action + 3) % 4;
          grid.message = 'Slippery tile changed your direction.';
        }
        const deltas = [[-1,0],[0,1],[1,0],[0,-1]];
        const d = deltas[action];
        grid.dir = action;
        const nr = clamp(grid.pos[0] + d[0], 0, cfg.rows - 1);
        const nc = clamp(grid.pos[1] + d[1], 0, cfg.cols - 1);
        if (!grid.walls.has(nr + ',' + nc)) grid.pos = [nr, nc];
        grid.steps += 1;
        grid.score += cfg.stepReward;
        const here = keyOf(grid.pos);
        grid.visited.add(here);
        if (grid.portals.has(here)) {
          grid.pos = grid.portals.get(here).slice();
          grid.message = 'Warp portal activated.';
          screenEffect('reward');
        }
        const afterPortal = keyOf(grid.pos);
        if (grid.traps.has(afterPortal)) {
          grid.score += cfg.trapRewards[afterPortal] || 0;
          const trapCenter = cellCenter(grid.pos);
          emitBurst(trapCenter.x, trapCenter.y, '#fb7185', 22);
          grid.hits += 1;
          grid.message = cfg.kind === 'sarsa' ? 'Laser hit. Move carefully.' : 'Danger tile hit.';
          screenEffect('hit');
        }
        if (grid.bonusesSet.has(afterPortal) && !grid.bonuses.has(afterPortal)) {
          grid.bonuses.add(afterPortal);
          grid.score += 18;
          const bonusCenter = cellCenter(grid.pos);
          emitBurst(bonusCenter.x, bonusCenter.y, '#facc15', 22);
          grid.message = 'Bonus collected.';
          screenEffect('reward');
        }
        for (let i = 0; i < grid.keys.length; i++) {
          if (keyOf(grid.keys[i]) === afterPortal && !grid.collected.has(i)) {
            grid.collected.add(i);
            grid.score += cfg.keyReward;
            const coreCenter = cellCenter(grid.pos);
            emitBurst(coreCenter.x, coreCenter.y, '#2dd4bf', 28);
            grid.message = cfg.kind === 'sarsa' ? 'BOX condition completed. Now open SAFE.' : 'CORE activated.';
            screenEffect('reward');
          }
        }
        if (cfg.kind === 'dp' && !cfg.replay) {
          enemyCollision();
        } else {
          grid.phase += 1;
          for (const g of guardPositions()) {
            if (g[0] === grid.pos[0] && g[1] === grid.pos[1]) {
              grid.score += cfg.guardReward;
              const guardCenter = cellCenter(grid.pos);
              emitBurst(guardCenter.x, guardCenter.y, '#f97316', 26);
              grid.hits += 1;
              grid.pos = cfg.start.slice();
              grid.message = 'Security caught you. Back to start.';
              screenEffect('hit');
            }
          }
        }
        if (keyOf(grid.pos) === keyOf(cfg.goal)) {
          const requirementMet = cfg.kind === 'dp' || grid.collected.size >= grid.keys.length;
          if (requirementMet) {
            grid.score += cfg.goalReward;
            grid.won = true;
            const goalCenter = cellCenter(grid.pos);
            emitBurst(goalCenter.x, goalCenter.y, '#86efac', 46);
            grid.message = 'Room cleared. Great run.';
            screenEffect('win');
          } else {
            grid.score += cfg.blockedGoalPenalty;
            grid.message = 'The exit is locked. Collect the required item first.';
          }
        }
        animateGridMove(oldPos, grid.pos);
        draw();
      }
      const gridLayout = { cell: 53, bx: 145, by: 48 };
      function cellCenter(pos) {
        return {
          x: gridLayout.bx + pos[1] * gridLayout.cell + gridLayout.cell / 2,
          y: gridLayout.by + pos[0] * gridLayout.cell + gridLayout.cell / 2
        };
      }
      function visualPlayerCenter() {
        if (!grid || !grid.visualFrom || !grid.visualTo) return cellCenter(grid.pos);
        const elapsed = performance.now() - grid.moveStarted;
        const raw = clamp(elapsed / 115, 0, 1);
        const eased = 1 - Math.pow(1 - raw, 3);
        const row = grid.visualFrom[0] + (grid.visualTo[0] - grid.visualFrom[0]) * eased;
        const col = grid.visualFrom[1] + (grid.visualTo[1] - grid.visualFrom[1]) * eased;
        return {
          x: gridLayout.bx + col * gridLayout.cell + gridLayout.cell / 2,
          y: gridLayout.by + row * gridLayout.cell + gridLayout.cell / 2
        };
      }
      function visualGhostCenter(index, fallback) {
        if (cfg.replay || !grid.ghostVisualFrom[index] || !grid.ghostVisualTo[index]) {
          return cellCenter(fallback);
        }
        const elapsed = performance.now() - grid.ghostMoveStarted;
        const raw = clamp(elapsed / 190, 0, 1);
        const eased = raw * raw * (3 - 2 * raw);
        const from = grid.ghostVisualFrom[index];
        const to = grid.ghostVisualTo[index];
        const row = from[0] + (to[0] - from[0]) * eased;
        const col = from[1] + (to[1] - from[1]) * eased;
        return {
          x: gridLayout.bx + col * gridLayout.cell + gridLayout.cell / 2,
          y: gridLayout.by + row * gridLayout.cell + gridLayout.cell / 2
        };
      }
      function cellRect(pos) {
        return {
          x: gridLayout.bx + pos[1] * gridLayout.cell,
          y: gridLayout.by + pos[0] * gridLayout.cell,
          s: gridLayout.cell
        };
      }
      function writeLabel(text, x, y, size=11, color='#f8fafc') {
        ctx.fillStyle = color;
        ctx.font = '900 ' + size + 'px Segoe UI';
        ctx.textAlign = 'center';
        ctx.fillText(text, x, y);
        ctx.textAlign = 'left';
      }
      function gridBack(title, subtitle, colorA, colorB) {
        drawArtBase(.34);
        const g = ctx.createLinearGradient(0, 0, W, 56);
        g.addColorStop(0, colorA);
        g.addColorStop(1, 'rgba(2,6,23,.35)');
        ctx.fillStyle = g;
        ctx.fillRect(0,0,W,56);
        ctx.fillStyle = '#f8fafc';
        ctx.font = '900 22px Segoe UI';
        ctx.fillText(title, 42, 28);
        ctx.fillStyle = 'rgba(226,232,240,.78)';
        ctx.font = '800 12px Segoe UI';
        ctx.fillText(subtitle, 250, 28);
      }
      function drawPacAgent(px, py) {
        const rotations = [-Math.PI / 2, 0, Math.PI / 2, Math.PI];
        const jaw = .16 + Math.abs(Math.sin(grid.anim * .014)) * .34;
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(rotations[grid.dir] || 0);
        ctx.shadowColor = 'rgba(250,204,21,.72)';
        ctx.shadowBlur = 14;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.arc(0, 0, 20, jaw, Math.PI * 2 - jaw);
        ctx.closePath();
        ctx.fillStyle = cfg.colors.agent;
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#fde047';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,.65)';
        ctx.beginPath();
        ctx.arc(-6, -10, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#111827';
        ctx.beginPath();
        ctx.arc(-5, -10, 2.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = 'rgba(255,255,255,.24)';
        ctx.beginPath();
        ctx.arc(-7, -7, 5, Math.PI * 1.1, Math.PI * 1.7);
        ctx.fill();
        ctx.restore();
      }
      function drawGhost(px, py, color='#fb7185', lookX=0, lookY=1, index=0) {
        const length = Math.hypot(lookX, lookY) || 1;
        const eyeX = clamp(lookX / length * 2.6, -2.6, 2.6);
        const eyeY = clamp(lookY / length * 2.6, -2.6, 2.6);
        ctx.save();
        ctx.translate(px, py);
        ctx.shadowColor = color;
        ctx.shadowBlur = 13;
        const body = ctx.createLinearGradient(0, -24, 0, 18);
        body.addColorStop(0, color);
        body.addColorStop(1, index % 2 ? '#0891b2' : '#e11d48');
        ctx.fillStyle = body;
        ctx.beginPath();
        ctx.moveTo(-18, 17);
        ctx.lineTo(-18, -3);
        ctx.arc(0, -3, 18, Math.PI, 0);
        ctx.lineTo(18, 17);
        ctx.quadraticCurveTo(13, 10, 8, 17);
        ctx.quadraticCurveTo(3, 10, -2, 17);
        ctx.quadraticCurveTo(-7, 10, -12, 17);
        ctx.quadraticCurveTo(-15, 13, -18, 17);
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = 'rgba(255,255,255,.38)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.ellipse(-7, -7, 6, 7, 0, 0, Math.PI * 2);
        ctx.ellipse(7, -7, 6, 7, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = '#172554';
        ctx.beginPath();
        ctx.arc(-7 + eyeX, -7 + eyeY, 2.8, 0, Math.PI * 2);
        ctx.arc(7 + eyeX, -7 + eyeY, 2.8, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = 'rgba(15,23,42,.75)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(0, 6, 6, .2, Math.PI - .2);
        ctx.stroke();
        ctx.restore();
      }
      function drawVaultRunner(px, py) {
        const reach = grid.dir === 1 ? 1 : (grid.dir === 3 ? -1 : 0);
        ctx.save();
        ctx.translate(px, py);
        ctx.fillStyle = 'rgba(2,6,23,.45)';
        ctx.beginPath(); ctx.ellipse(0, 20, 20, 7, 0, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#312e81';
        ctx.lineWidth = 7;
        ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(-7, 11); ctx.lineTo(-9, 22); ctx.moveTo(7, 11); ctx.lineTo(10, 22); ctx.stroke();
        rr(-14, -4, 28, 24, 6, '#2563eb', '#bfdbfe');
        ctx.fillStyle = '#f59e0b';
        ctx.fillRect(-3, -2, 6, 19);
        ctx.strokeStyle = '#fde68a';
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(-12, 2);
        ctx.lineTo(reach < 0 ? -25 : -18, reach < 0 ? -2 : 10);
        ctx.moveTo(12, 2);
        ctx.lineTo(reach > 0 ? 25 : 18, reach > 0 ? -2 : 10);
        ctx.stroke();
        rr(-12, -18, 24, 17, 7, '#fed7aa', '#78350f');
        ctx.fillStyle = '#111827';
        ctx.fillRect(-8, -13, 16, 6);
        ctx.fillStyle = '#38bdf8';
        ctx.fillRect(-6, -12, 4, 3);
        ctx.fillRect(2, -12, 4, 3);
        ctx.fillStyle = '#facc15';
        ctx.beginPath();
        ctx.arc(0, -18, 14, Math.PI, 0);
        ctx.lineTo(13, -14);
        ctx.lineTo(-13, -14);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = '#fef3c7';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#e0f2fe';
        ctx.fillRect(-4, -24, 8, 5);
        ctx.restore();
      }
      function drawBomberman(px, py) {
        ctx.save();
        ctx.translate(px, py);
        ctx.fillStyle = 'rgba(2,6,23,.5)';
        ctx.beginPath(); ctx.ellipse(0, 21, 21, 7, 0, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#db2777';
        ctx.lineWidth = 8;
        ctx.lineCap = 'round';
        ctx.beginPath(); ctx.moveTo(-8, 11); ctx.lineTo(-12, 22); ctx.moveTo(8, 11); ctx.lineTo(13, 22); ctx.stroke();
        rr(-14, 0, 28, 22, 8, '#2563eb', '#bfdbfe');
        ctx.strokeStyle = '#f9a8d4';
        ctx.lineWidth = 7;
        ctx.beginPath(); ctx.moveTo(-12, 5); ctx.lineTo(-23, 11); ctx.moveTo(12, 5); ctx.lineTo(23, 11); ctx.stroke();
        ctx.fillStyle = '#ffffff';
        ctx.beginPath(); ctx.arc(0, -10, 19, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = '#172554';
        ctx.beginPath(); ctx.ellipse(0, -8, 12, 13, 0, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(-6, -13, 3, 7);
        ctx.fillRect(4, -13, 3, 7);
        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(0, -28); ctx.lineTo(8, -37); ctx.stroke();
        ctx.fillStyle = '#f472b6';
        ctx.beginPath(); ctx.arc(10, -39, 6, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fbcfe8'; ctx.lineWidth = 2; ctx.stroke();
        ctx.restore();
      }
      function drawIceLab() {
        const cell = gridLayout.cell, bx = gridLayout.bx, by = gridLayout.by;
        gridBack('Pac-Man Ice Maze', 'known model / moving ghosts / slippery tiles', '#020617', '#082f49');
        for (let r = 0; r < cfg.rows; r++) {
          for (let c = 0; c < cfg.cols; c++) {
            const k = r + ',' + c, x = bx + c * cell, y = by + r * cell;
            let fill = grid.visited.has(k) ? 'rgba(18,55,101,.88)' : 'rgba(7,17,31,.78)';
            let stroke = 'rgba(148,163,184,.18)';
            if (grid.walls.has(k)) { fill = 'rgba(30,58,138,.94)'; stroke = '#38bdf8'; }
            rr(x + 2, y + 2, cell - 4, cell - 4, 9, fill, stroke);
            if (grid.slippery.has(k)) {
              ctx.strokeStyle = '#e0f2fe'; ctx.lineWidth = 2;
              for (let i = 0; i < 3; i++) { ctx.beginPath(); ctx.moveTo(x+10, y+14+i*12); ctx.lineTo(x+cell-10, y+7+i*12); ctx.stroke(); }
              writeLabel('ICE', x + cell / 2, y + cell - 11, 10, '#082f49');
            }
            if (grid.traps.has(k)) {
              ctx.strokeStyle = '#fb7185'; ctx.lineWidth = 3;
              ctx.beginPath();
              ctx.moveTo(x+8,y+15); ctx.lineTo(x+22,y+24); ctx.lineTo(x+15,y+38); ctx.lineTo(x+40,y+45);
              ctx.stroke();
              writeLabel('CRACK', x+cell/2, y+cell-9, 8, '#fecdd3');
            }
          }
        }
        drawGridObjects('ice');
        const p = visualPlayerCenter();
        const ghostColors = ['#fb7185', '#22d3ee', '#f97316'];
        for (const [index, enemy] of guardPositions().entries()) {
          const ghost = visualGhostCenter(index, enemy);
          const bob = Math.sin(grid.anim * .008 + index) * 2;
          drawGhost(
            ghost.x,
            ghost.y + bob,
            ghostColors[index % ghostColors.length],
            p.x - ghost.x,
            p.y - ghost.y,
            index
          );
        }
        drawPacAgent(p.x, p.y);
        writeLabel('GHOST HUNT ' + grid.phase, 720, 28, 12, '#bae6fd');
      }
      function drawMuseumVault() {
        const cell = gridLayout.cell, bx = gridLayout.bx, by = gridLayout.by;
        gridBack('Sokoban Vault', 'push BOX onto TARGET, then enter SAFE', '#120c24', '#2e1065');
        for (let r = 0; r < cfg.rows; r++) {
          for (let c = 0; c < cfg.cols; c++) {
            const k = r + ',' + c, x = bx + c * cell, y = by + r * cell;
            let fill = (r + c) % 2 ? 'rgba(30,21,53,.78)' : 'rgba(22,15,42,.78)';
            if (grid.visited.has(k)) fill = 'rgba(46,35,99,.90)';
            let stroke = 'rgba(167,139,250,.16)';
            if (grid.walls.has(k)) { fill = 'rgba(76,29,149,.94)'; stroke = '#c4b5fd'; }
            rr(x + 3, y + 3, cell - 6, cell - 6, 4, fill, stroke);
            if (grid.walls.has(k)) {
              ctx.fillStyle = 'rgba(233,213,255,.28)';
              for (let i = 0; i < 3; i++) ctx.fillRect(x + 8, y + 10 + i * 13, cell - 16, 5);
            }
            if (grid.slippery.has(k)) {
              ctx.beginPath(); ctx.ellipse(x + cell/2, y + cell/2, 17, 11, -0.4, 0, Math.PI*2);
              ctx.fillStyle = 'rgba(168,85,247,.42)'; ctx.fill();
              writeLabel('OIL', x+cell/2, y+cell/2+4, 10, '#e9d5ff');
            }
            if (grid.traps.has(k)) {
              const shift = Math.sin(grid.anim * .008 + r) * 10;
              ctx.strokeStyle = '#fb7185'; ctx.lineWidth = 4;
              ctx.shadowColor = '#fb7185'; ctx.shadowBlur = 10;
              ctx.beginPath(); ctx.moveTo(x+8, y+cell/2+shift); ctx.lineTo(x+cell-8, y+cell/2-shift); ctx.stroke();
              ctx.shadowBlur = 0;
            }
          }
        }
        for (const target of grid.targets || []) {
          const r = cellRect(target), cx = r.x + r.s / 2, cy = r.y + r.s / 2;
          rr(r.x + 10, r.y + 10, r.s - 20, r.s - 20, 8, 'rgba(34,197,94,.16)', '#86efac');
          ctx.strokeStyle = '#bbf7d0'; ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(cx - 15, cy); ctx.lineTo(cx + 15, cy); ctx.moveTo(cx, cy - 15); ctx.lineTo(cx, cy + 15); ctx.stroke();
          writeLabel('TARGET', cx, cy + 23, 8, '#dcfce7');
        }
        for (const box of grid.boxes || []) {
          const r = cellRect(box), cx = r.x + r.s / 2, cy = r.y + r.s / 2;
          const onTarget = (grid.targets || []).some(target => keyOf(target) === keyOf(box));
          rr(r.x + 8, r.y + 8, r.s - 16, r.s - 16, 6, onTarget ? '#65a30d' : '#d97706', onTarget ? '#bbf7d0' : '#fde68a');
          ctx.strokeStyle = onTarget ? '#dcfce7' : '#78350f';
          ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(r.x + 13, r.y + 13); ctx.lineTo(r.x + r.s - 13, r.y + r.s - 13); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(r.x + r.s - 13, r.y + 13); ctx.lineTo(r.x + 13, r.y + r.s - 13); ctx.stroke();
          writeLabel('BOX', cx, cy + 4, 11, '#111827');
        }
        for (const b of cfg.bonuses || []) if (!grid.bonuses.has(keyOf(b))) {
          const r = cellRect(b), cx = r.x + r.s / 2, cy = r.y + r.s / 2;
          ctx.beginPath(); ctx.arc(cx, cy, 15, 0, Math.PI * 2); ctx.fillStyle = '#facc15'; ctx.fill();
          writeLabel('COIN', cx, cy + 4, 8, '#111827');
        }
        const safe = cellRect(cfg.goal);
        rr(safe.x + 6, safe.y + 6, safe.s - 12, safe.s - 12, 10, '#16a34a', '#bbf7d0');
        ctx.beginPath(); ctx.arc(safe.x + safe.s / 2, safe.y + safe.s / 2, 10, 0, Math.PI * 2);
        ctx.strokeStyle = '#052e16'; ctx.lineWidth = 4; ctx.stroke();
        writeLabel('SAFE', safe.x + safe.s / 2, safe.y + safe.s - 12, 10, '#dcfce7');
        const p = visualPlayerCenter();
        drawVaultRunner(p.x, p.y);
        const scannerX = bx + 30 + ((grid.anim * .08) % (cell * 9));
        const scannerY = by + cell * 4.45;
        ctx.beginPath(); ctx.arc(scannerX, scannerY, 12, 0, Math.PI*2);
        ctx.fillStyle = '#fb7185'; ctx.fill();
        ctx.strokeStyle = '#fecdd3'; ctx.lineWidth = 3; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(scannerX, scannerY+12); ctx.lineTo(scannerX, scannerY+38);
        ctx.strokeStyle = 'rgba(251,113,133,.5)'; ctx.lineWidth = 5; ctx.stroke();
      }
      function drawReactorCore() {
        const cell = gridLayout.cell, bx = gridLayout.bx, by = gridLayout.by;
        gridBack('Bomberman Reactor', 'bomb maze / cores / warp tunnels', '#02130f', '#042f2e');
        for (let r = 0; r < cfg.rows; r++) {
          for (let c = 0; c < cfg.cols; c++) {
            const k = r + ',' + c, x = bx + c * cell, y = by + r * cell;
            let fill = grid.visited.has(k) ? 'rgba(18,61,55,.90)' : ((r + c) % 2 ? 'rgba(5,34,29,.78)' : 'rgba(4,20,17,.78)');
            let stroke = 'rgba(45,212,191,.14)';
            if (grid.walls.has(k)) { fill = 'rgba(15,118,110,.94)'; stroke = '#99f6e4'; }
            rr(x + 2, y + 2, cell - 4, cell - 4, 6, fill, stroke);
            if (grid.walls.has(k)) {
              ctx.fillStyle = 'rgba(204,251,241,.22)';
              ctx.fillRect(x + 8, y + 11, cell - 16, 7);
              ctx.fillRect(x + 14, y + 29, cell - 22, 7);
            } else {
              ctx.strokeStyle = 'rgba(45,212,191,.08)'; ctx.lineWidth = 1;
              ctx.beginPath(); ctx.moveTo(x+cell/2, y+7); ctx.lineTo(x+cell/2, y+cell-7); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(x+7, y+cell/2); ctx.lineTo(x+cell-7, y+cell/2); ctx.stroke();
            }
            if (grid.slippery.has(k)) {
              ctx.beginPath(); ctx.arc(x+cell/2, y+cell/2, 18, 0, Math.PI*2);
              ctx.fillStyle = 'rgba(20,184,166,.45)'; ctx.fill();
              writeLabel('SLIME', x+cell/2, y+cell/2+4, 8, '#ccfbf1');
            }
            if (grid.traps.has(k)) {
              const cx = x + cell / 2, cy = y + cell / 2;
              const blast = 17 + Math.sin(grid.anim * .012 + r + c) * 8;
              ctx.strokeStyle = 'rgba(251,146,60,.72)';
              ctx.lineWidth = 6;
              ctx.beginPath(); ctx.moveTo(cx - blast, cy); ctx.lineTo(cx + blast, cy); ctx.stroke();
              ctx.beginPath(); ctx.moveTo(cx, cy - blast); ctx.lineTo(cx, cy + blast); ctx.stroke();
              ctx.fillStyle = '#111827';
              ctx.beginPath(); ctx.arc(cx, cy, 15, 0, Math.PI * 2); ctx.fill();
              ctx.strokeStyle = '#fed7aa'; ctx.lineWidth = 3; ctx.stroke();
              ctx.strokeStyle = '#facc15'; ctx.lineWidth = 3;
              ctx.beginPath(); ctx.moveTo(cx + 8, cy - 12); ctx.quadraticCurveTo(cx + 23, cy - 26, cx + 12, cy - 31); ctx.stroke();
              ctx.fillStyle = '#fb923c';
              ctx.beginPath();
              for (let i = 0; i < 8; i++) {
                const a = i * Math.PI / 4;
                const rad = i % 2 ? 5 : 10;
                const px = cx + 23 + Math.cos(a) * rad;
                const py = cy - 27 + Math.sin(a) * rad;
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
              }
              ctx.closePath(); ctx.fill();
            }
          }
        }
        drawGridObjects('reactor');
        const p = visualPlayerCenter();
        drawBomberman(p.x, p.y);
      }
      function drawGridObjects(style) {
        function badge(pos, text, fill, color='#f8fafc') {
          const r = cellRect(pos), cx = r.x + r.s/2, cy = r.y + r.s/2;
          if (style === 'reactor' && text.startsWith('C')) {
            ctx.beginPath(); ctx.arc(cx, cy, 19, 0, Math.PI * 2);
            ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = '#fef9c3'; ctx.lineWidth = 3; ctx.stroke();
            ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fillStyle = '#0f172a'; ctx.fill();
          } else if (style === 'museum' && text === (cfg.labels.key || 'KEY')) {
            rr(cx-20, cy-13, 40, 26, 5, fill, '#fef9c3');
            ctx.fillStyle = '#111827'; ctx.fillRect(cx-14, cy-4, 28, 3);
          } else {
            rr(r.x+8, r.y+8, r.s-16, r.s-16, 12, fill, '#f8fafc');
          }
          writeLabel(text, cx, cy+4, style === 'reactor' ? 10 : 11, color);
        }
        badge(cfg.goal, cfg.goalLabel, '#16a34a');
        for (let i = 0; i < grid.keys.length; i++) {
          if (!grid.collected.has(i)) badge(grid.keys[i], cfg.kind === 'q_learning' ? 'C' + (i+1) : (cfg.labels.key || 'KEY'), cfg.colors.key, '#111827');
        }
        for (const b of cfg.bonuses || []) if (!grid.bonuses.has(keyOf(b))) badge(b, cfg.labels.bonus || 'BON', '#f59e0b', '#111827');
        for (const p of cfg.portals || []) {
          const r = cellRect(p.from), cx = r.x+r.s/2, cy = r.y+r.s/2;
          ctx.beginPath(); ctx.arc(cx, cy, 21, 0, Math.PI*2); ctx.strokeStyle = cfg.colors.portal; ctx.lineWidth = 5; ctx.stroke();
          ctx.beginPath(); ctx.arc(cx, cy, 11, 0, Math.PI*2); ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 3; ctx.stroke();
          writeLabel(cfg.labels.portal || 'WARP', cx, cy+4, 8);
        }
        for (const g of guardPositions()) {
          const r = cellRect(g), cx = r.x+r.s/2, cy = r.y+r.s/2;
          if (style === 'reactor') {
            ctx.save();
            ctx.translate(cx, cy);
            ctx.shadowColor = '#f97316';
            ctx.shadowBlur = 12;
            rr(-19, -13, 38, 26, 9, '#c2410c', '#fed7aa');
            ctx.shadowBlur = 0;
            rr(-12, -8, 24, 11, 4, '#172554', '#7dd3fc');
            ctx.fillStyle = '#67e8f9';
            ctx.fillRect(-7, -5, 4, 4);
            ctx.fillRect(3, -5, 4, 4);
            ctx.strokeStyle = '#fdba74';
            ctx.lineWidth = 4;
            ctx.lineCap = 'round';
            ctx.beginPath();
            ctx.moveTo(-18, 8); ctx.lineTo(-24, 15);
            ctx.moveTo(18, 8); ctx.lineTo(24, 15);
            ctx.stroke();
            ctx.strokeStyle = '#fde68a';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.moveTo(0, -13); ctx.lineTo(0, -22); ctx.stroke();
            ctx.beginPath(); ctx.arc(0, -24, 3, 0, Math.PI*2);
            ctx.fillStyle = '#facc15'; ctx.fill();
            ctx.restore();
          } else {
            rr(cx-18, cy-16, 36, 32, 10, '#ea580c', '#fed7aa');
            writeLabel(cfg.labels.guard || 'SEC', cx, cy+4, 10);
          }
        }
      }
      function drawGrid() {
        if (cfg.kind === 'sarsa') drawMuseumVault();
        else if (cfg.kind === 'q_learning') drawReactorCore();
        else drawIceLab();
        updateStats(grid.score, grid.steps, grid.won ? 'WON' : 'PLAYING', grid.message);
      }

      let cont = null;
      const input = {vx: 0, vy: 0, left:false, right:false, up:false, down:false};
      function rand(min,max){ return min + Math.random() * (max-min); }
      function initContinuous() {
        cont = {
          x: cfg.start[0], y: cfg.start[1], vx: 0, vy: 0,
          score: 0, steps: 0, won: false, message: 'Choose a discrete velocity and reach ' + cfg.goalLabel + ' quickly.',
          path: [], obstacles: [], teleports: [], meteors: [], cooldown: 0, effectCooldown: 0, hits: 0
        };
        if (cfg.teleportMode) {
          for (let i = 0; i < cfg.teleportCount; i++) {
            cont.teleports.push({x: rand(1.2, 8.8), y: rand(1.2, 8.8), pulse: Math.random() * Math.PI * 2});
          }
          cont.message = 'Step into a portal to jump to a random location. Use jumps to reach EXIT.';
        } else if (cfg.obstacleMode) {
          for (let i = 0; i < cfg.obstacleCount; i++) {
            const axis = Math.random() < .5 ? 0 : 1;
            const direction = Math.random() < .5 ? -1 : 1;
            cont.obstacles.push({x: rand(1.4, 8.6), y: rand(1.4, 8.6), axis, direction, dx: axis===0?direction:0, dy: axis===1?direction:0});
          }
          cont.message = 'Portal hazards move. Observe forward and reach EXIT.';
        } else {
          for (const [zone, h] of cfg.hazards.entries()) {
            const horizontal = (h[2] - h[0]) >= (h[3] - h[1]);
            const span = horizontal ? h[2] - h[0] : h[3] - h[1];
            const count = Math.max(3, Math.min(5, Math.round(span)));
            for (let i = 0; i < count; i++) {
              const progress = (i + .5) / count;
              const direction = (i + zone) % 2 === 0 ? 1 : -1;
              cont.meteors.push({
                zone,
                x: horizontal ? h[0] + (h[2] - h[0]) * progress : (h[0] + h[2]) / 2 + (i % 2 ? .10 : -.10),
                y: horizontal ? (h[1] + h[3]) / 2 + (i % 2 ? .08 : -.08) : h[1] + (h[3] - h[1]) * progress,
                dx: horizontal ? direction * (.007 + i * .0015) : 0,
                dy: horizontal ? 0 : direction * (.007 + i * .0012),
                r: .22 + (i % 3) * .025,
                spin: rand(0, Math.PI * 2),
                spinRate: direction * (.022 + i * .004),
                seed: zone * 11 + i * 7 + 3
              });
            }
          }
          cont.message = 'Avoid the asteroid fields and reach PAD using one of nine discrete velocities.';
        }
      }
      function setVector(vx, vy) { input.vx = vx; input.vy = vy; focusGame(); }
      function inHazard(x,y) {
        return cfg.hazards.some(h => x >= h[0] && x <= h[2] && y >= h[1] && y <= h[3]);
      }
      function updateLanderMeteors(scale=1) {
        if (cfg.obstacleMode || cfg.teleportMode) return;
        for (const meteor of cont.meteors || []) {
          const h = cfg.hazards[meteor.zone];
          meteor.x += meteor.dx * scale;
          meteor.y += meteor.dy * scale;
          meteor.spin += meteor.spinRate * scale;
          const margin = meteor.r * .7;
          if (meteor.x < h[0] + margin || meteor.x > h[2] - margin) {
            meteor.dx *= -1;
            meteor.x = clamp(meteor.x, h[0] + margin, h[2] - margin);
          }
          if (meteor.y < h[1] + margin || meteor.y > h[3] - margin) {
            meteor.dy *= -1;
            meteor.y = clamp(meteor.y, h[1] + margin, h[3] - margin);
          }
        }
      }
      function continuousCanvasPoint(x, y) {
        if (cfg.teleportMode) {
          return {x: 55 + x / cfg.roomSize * 610, y: 70 + 470 - y / cfg.roomSize * 470};
        }
        const map = arenaMap();
        return {x: map.toX(x), y: map.toY(y)};
      }
      function stepContinuous() {
        if (!cont || cont.won) return;
        cont.effectCooldown = Math.max(0, cont.effectCooldown - 1);
        updateLanderMeteors();
        let vx = input.vx || ((input.right?1:0) - (input.left?1:0));
        let vy = input.vy || ((input.up?1:0) - (input.down?1:0));
        cont.vx = vx; cont.vy = vy;
        const oldX = cont.x, oldY = cont.y;
        const oldDist = Math.hypot(cfg.goal[0]-cont.x, cfg.goal[1]-cont.y);
        const speed = 0.02;
        const rawX = cont.x + vx * speed, rawY = cont.y + vy * speed;
        if (rawX < 0 || rawX > cfg.roomSize || rawY < 0 || rawY > cfg.roomSize) cont.score += cfg.wallPenalty;
        cont.x = clamp(rawX, 0, cfg.roomSize);
        cont.y = clamp(rawY, 0, cfg.roomSize);
        cont.steps += 1;
        if (inHazard(cont.x, cont.y)) {
          cont.score += cfg.hazardPenalty;
          const collision = continuousCanvasPoint(cont.x, cont.y);
          cont.x = oldX;
          cont.y = oldY;
          cont.vx = 0;
          cont.vy = 0;
          cont.message = 'Asteroid impact. Velocity reset; choose a route around the field.';
          if (cont.effectCooldown === 0) {
            cont.hits += 1;
            cont.effectCooldown = 30;
            emitBurst(collision.x, collision.y, '#fb923c', 34);
            screenEffect('hit');
          }
        }
        const newDist = Math.hypot(cfg.goal[0]-cont.x, cfg.goal[1]-cont.y);
        cont.score += cfg.stepReward + cfg.progressScale * (oldDist - newDist);
        if (cfg.obstacleMode) {
          const half = cfg.obstacleWidth / 2;
          for (const o of cont.obstacles) {
            o.x += o.dx * 0.01; o.y += o.dy * 0.01;
            if (o.x < .7 || o.x > 9.3) o.dx *= -1;
            if (o.y < .7 || o.y > 9.3) o.dy *= -1;
            if (Math.abs(cont.x-o.x) < half && Math.abs(cont.y-o.y) < half) {
              const collision = continuousCanvasPoint(cont.x, cont.y);
              emitBurst(collision.x, collision.y, '#f97316', 30);
              const collisionDist = Math.hypot(cfg.goal[0]-cont.x, cfg.goal[1]-cont.y);
              cont.score += cfg.obstaclePenalty; cont.x = cfg.start[0]; cont.y = cfg.start[1];
              cont.vx = 0; cont.vy = 0;
              const resetDist = Math.hypot(cfg.goal[0]-cont.x, cfg.goal[1]-cont.y);
              cont.score += cfg.progressScale * (collisionDist - resetDist);
              cont.message = 'Portal collision. Teleported back to start.';
              cont.hits += 1;
              cont.effectCooldown = 30;
              screenEffect('hit');
            }
          }
        }
        if (cfg.teleportMode) {
          cont.cooldown = Math.max(0, cont.cooldown - 1);
          if (cont.steps % 180 === 0) {
            for (const t of cont.teleports) { t.x = rand(1.0, 9.0); t.y = rand(1.0, 9.0); }
            cont.message = 'Portal gates reshuffled.';
          }
          for (const t of cont.teleports) {
            if (cont.cooldown === 0 && Math.hypot(cont.x - t.x, cont.y - t.y) < 0.38) {
              if (Math.random() < 0.72 && cont.teleports.length > 1) {
                const other = cont.teleports[Math.floor(Math.random() * cont.teleports.length)];
                cont.x = clamp(other.x + rand(-0.35, 0.35), 0, cfg.roomSize);
                cont.y = clamp(other.y + rand(-0.35, 0.35), 0, cfg.roomSize);
                cont.message = 'Teleport jump! You landed near another gate.';
              } else {
                cont.x = rand(0.6, 9.4);
                cont.y = rand(0.6, 9.4);
                cont.message = 'Unstable teleport! Random landing point.';
              }
              const arrival = continuousCanvasPoint(cont.x, cont.y);
              emitBurst(arrival.x, arrival.y, '#c084fc', 34);
              cont.score += 7;
              cont.cooldown = 45;
              screenEffect('reward');
              break;
            }
          }
        }
        const dist = Math.hypot(cfg.goal[0]-cont.x, cfg.goal[1]-cont.y);
        if (dist <= cfg.goalRadius) {
          cont.score += cfg.goalReward;
          cont.won = true;
          const goalBurst = continuousCanvasPoint(cont.x, cont.y);
          emitBurst(goalBurst.x, goalBurst.y, '#86efac', 52);
          cont.message = cfg.teleportMode ? 'Portal exit reached. Room cleared.' : 'Touchdown complete. Room cleared.';
          screenEffect('win');
        }
        if (cont.steps % 3 === 0) cont.path.push([cont.x, cont.y]);
      }
      function arenaMap() {
        const bx = 150, by = 48, bw = 520, bh = 520;
        return {
          bx, by, bw, bh,
          toX: x => bx + x / cfg.roomSize * bw,
          toY: y => by + bh - y / cfg.roomSize * bh
        };
      }
      function animTick() {
        return cont && cont.anim ? cont.anim / 16.67 : (cont ? cont.steps : 0);
      }
      function drawDrone(px, py) {
        ctx.save();
        ctx.translate(px, py);
        const tilt = (cont.vx || 0) * 0.16;
        ctx.rotate(tilt);
        ctx.fillStyle = 'rgba(2,6,23,.55)';
        ctx.beginPath(); ctx.ellipse(0, 31, 30, 8, 0, 0, Math.PI * 2); ctx.fill();
        const flame = 34 + Math.sin(animTick() * .55) * 7;
        if (cont.vx !== 0 || cont.vy !== 0) {
          const thrust = ctx.createLinearGradient(0, 20, 0, flame + 15);
          thrust.addColorStop(0, '#fef08a');
          thrust.addColorStop(.45, '#fb923c');
          thrust.addColorStop(1, 'rgba(239,68,68,0)');
          ctx.fillStyle = thrust;
          ctx.beginPath();
          ctx.moveTo(-11, 19);
          ctx.quadraticCurveTo(0, flame + 18, 11, 19);
          ctx.closePath();
          ctx.fill();
        }
        ctx.shadowColor = '#38bdf8';
        ctx.shadowBlur = 14;
        const hull = ctx.createLinearGradient(-20, -25, 20, 22);
        hull.addColorStop(0, '#f8fafc');
        hull.addColorStop(.52, '#bae6fd');
        hull.addColorStop(1, '#2563eb');
        ctx.fillStyle = hull;
        ctx.beginPath();
        ctx.moveTo(0, -30);
        ctx.quadraticCurveTo(20, -14, 23, 18);
        ctx.lineTo(10, 23);
        ctx.lineTo(-10, 23);
        ctx.quadraticCurveTo(-23, 7, -23, 18);
        ctx.quadraticCurveTo(-20, -14, 0, -30);
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#7dd3fc'; ctx.lineWidth = 3; ctx.stroke();
        const glass = ctx.createLinearGradient(0, -19, 0, 2);
        glass.addColorStop(0, '#67e8f9');
        glass.addColorStop(1, '#172554');
        ctx.fillStyle = glass;
        ctx.beginPath(); ctx.ellipse(0, -8, 10, 13, 0, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#e0f2fe'; ctx.lineWidth = 2; ctx.stroke();
        ctx.fillStyle = '#f97316';
        ctx.fillRect(-16, 7, 7, 12);
        ctx.fillRect(9, 7, 7, 12);
        ctx.strokeStyle = '#e0f2fe'; ctx.lineWidth = 4;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(-15, 19); ctx.lineTo(-27, 31); ctx.lineTo(-32, 31);
        ctx.moveTo(15, 19); ctx.lineTo(27, 31); ctx.lineTo(32, 31);
        ctx.stroke();
        ctx.fillStyle = '#f8fafc';
        ctx.beginPath(); ctx.arc(-32, 31, 4, 0, Math.PI*2); ctx.arc(32, 31, 4, 0, Math.PI*2); ctx.fill();
        ctx.restore();
      }
      function drawPortalScout(px, py) {
        const moving = Math.abs(cont.vx || 0) + Math.abs(cont.vy || 0) > 0;
        const stride = moving ? Math.sin(animTick() * .34) * 8 : 0;
        const facing = (cont.vx || 1) < 0 ? -1 : 1;
        ctx.save();
        ctx.translate(px, py);
        ctx.scale(facing, 1);
        ctx.fillStyle = 'rgba(2,6,23,.58)';
        ctx.beginPath(); ctx.ellipse(0, 29, 27, 8, 0, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth = 9;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(-7, 13); ctx.lineTo(-9-stride, 28);
        ctx.moveTo(7, 13); ctx.lineTo(9+stride, 28);
        ctx.stroke();
        ctx.strokeStyle = '#fb923c';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(-9-stride, 28); ctx.lineTo(-18-stride, 29);
        ctx.moveTo(9+stride, 28); ctx.lineTo(18+stride, 29);
        ctx.stroke();
        rr(-15, -7, 30, 27, 7, '#e2e8f0', '#ffffff');
        ctx.fillStyle = '#f97316';
        ctx.fillRect(4, -5, 6, 22);
        rr(-21, -4, 8, 22, 3, '#334155', '#94a3b8');
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 7;
        ctx.beginPath();
        ctx.moveTo(-13, -1); ctx.lineTo(-23, 7 + stride * .35);
        ctx.moveTo(13, -1); ctx.lineTo(24, 4 - stride * .35);
        ctx.stroke();
        ctx.fillStyle = '#fb923c';
        ctx.beginPath(); ctx.arc(-24, 8 + stride * .35, 5, 0, Math.PI*2); ctx.arc(25, 4 - stride * .35, 5, 0, Math.PI*2); ctx.fill();
        ctx.shadowColor = '#22d3ee';
        ctx.shadowBlur = 10;
        ctx.fillStyle = '#f8fafc';
        ctx.beginPath(); ctx.arc(0, -19, 17, 0, Math.PI*2); ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 2; ctx.stroke();
        const visor = ctx.createLinearGradient(0, -28, 0, -12);
        visor.addColorStop(0, '#67e8f9');
        visor.addColorStop(1, '#172554');
        ctx.fillStyle = visor;
        ctx.beginPath();
        ctx.ellipse(4, -19, 12, 9, 0, -.85, .85);
        ctx.lineTo(-7, -12);
        ctx.quadraticCurveTo(-10, -20, -6, -26);
        ctx.closePath();
        ctx.fill();
        ctx.strokeStyle = '#a5f3fc'; ctx.lineWidth = 2; ctx.stroke();
        ctx.strokeStyle = '#f8fafc'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(-5, -35); ctx.lineTo(-1, -43); ctx.stroke();
        ctx.fillStyle = '#22d3ee';
        ctx.beginPath(); ctx.arc(0, -45, 4, 0, Math.PI*2); ctx.fill();
        ctx.restore();
      }
      function drawFlamingMeteor(meteor, map, index) {
        const mx = map.toX(meteor.x);
        const my = map.toY(meteor.y);
        const mr = meteor.r / cfg.roomSize * map.bw;
        const screenDx = meteor.dx;
        const screenDy = -meteor.dy;
        const heading = Math.atan2(screenDy, screenDx || .0001);
        const pulse = .88 + Math.sin(animTick() * .32 + index) * .12;

        ctx.save();
        ctx.translate(mx, my);
        ctx.rotate(heading);
        ctx.globalCompositeOperation = 'screen';
        const glow = ctx.createRadialGradient(-mr * 1.4, 0, 1, -mr * 1.4, 0, mr * 3.4);
        glow.addColorStop(0, 'rgba(254,240,138,.82)');
        glow.addColorStop(.35, 'rgba(249,115,22,.46)');
        glow.addColorStop(1, 'rgba(239,68,68,0)');
        ctx.fillStyle = glow;
        ctx.beginPath(); ctx.ellipse(-mr * 1.6, 0, mr * 3.5, mr * 1.45, 0, 0, Math.PI * 2); ctx.fill();
        const flame = ctx.createLinearGradient(-mr * 4.3, 0, mr * .2, 0);
        flame.addColorStop(0, 'rgba(239,68,68,0)');
        flame.addColorStop(.28, '#ef4444');
        flame.addColorStop(.62, '#f97316');
        flame.addColorStop(1, '#fef08a');
        ctx.fillStyle = flame;
        ctx.beginPath();
        ctx.moveTo(mr * .15, -mr * .62);
        ctx.quadraticCurveTo(-mr * 1.5, -mr * 1.25 * pulse, -mr * 4.2, -mr * .18);
        ctx.quadraticCurveTo(-mr * 2.8, 0, -mr * 4.6, mr * .3);
        ctx.quadraticCurveTo(-mr * 1.6, mr * 1.1 * pulse, mr * .15, mr * .58);
        ctx.closePath();
        ctx.fill();
        ctx.globalCompositeOperation = 'source-over';
        ctx.fillStyle = '#fbbf24';
        for (let ember = 0; ember < 4; ember++) {
          const travel = ((animTick() * (.13 + ember * .02) + meteor.seed * 7 + ember * 19) % 42);
          const ey = Math.sin(animTick() * .09 + ember * 2 + meteor.seed) * mr * .65;
          ctx.beginPath();
          ctx.arc(-mr * 1.1 - travel, ey, Math.max(1.5, mr * (.16 - ember * .018)), 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.restore();

        ctx.save();
        ctx.translate(mx, my);
        ctx.rotate(meteor.spin);
        ctx.shadowColor = '#fb923c';
        ctx.shadowBlur = 12;
        const rock = ctx.createRadialGradient(-mr * .35, -mr * .35, mr * .1, 0, 0, mr * 1.25);
        rock.addColorStop(0, '#fdba74');
        rock.addColorStop(.28, '#9a3412');
        rock.addColorStop(.7, '#431407');
        rock.addColorStop(1, '#1c0a04');
        ctx.fillStyle = rock;
        ctx.beginPath();
        for (let point = 0; point < 11; point++) {
          const angle = point / 11 * Math.PI * 2;
          const wobble = .80 + ((point * 17 + meteor.seed * 13) % 29) / 100;
          const x = Math.cos(angle) * mr * wobble;
          const y = Math.sin(angle) * mr * wobble;
          if (point === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = '#fb923c';
        ctx.lineWidth = 2.2;
        ctx.stroke();
        ctx.fillStyle = 'rgba(28,10,4,.78)';
        ctx.beginPath(); ctx.arc(-mr * .28, -mr * .18, mr * .23, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(mr * .34, mr * .25, mr * .17, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = '#fed7aa';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(-mr * .05, -mr * .55);
        ctx.lineTo(mr * .12, -mr * .16);
        ctx.lineTo(-mr * .08, mr * .12);
        ctx.stroke();
        ctx.restore();
      }
      function drawDroneLanding() {
        drawArtBase(.18);
        ctx.fillStyle = 'rgba(2,6,23,.66)';
        ctx.fillRect(0,0,W,48);
        ctx.fillStyle = '#e0f2fe';
        ctx.font = '900 22px Segoe UI';
        ctx.fillText('Lunar Lander Pad', 42, 28);
        ctx.fillStyle = 'rgba(147,197,253,.75)';
        ctx.font = '800 12px Segoe UI';
        ctx.fillText('continuous position / discrete velocity / landing pad', 270, 28);
        const m = arenaMap();
        rr(m.bx, m.by, m.bw, m.bh, 18, 'rgba(7,26,48,.44)', '#60a5fa');
        for (let i=0;i<55;i++) {
          const x = m.bx + ((i*97 + animTick()*0.2) % m.bw);
          const y = m.by + ((i*53) % m.bh);
          ctx.fillStyle = i % 5 === 0 ? 'rgba(147,197,253,.35)' : 'rgba(226,232,240,.18)';
          ctx.fillRect(x, y, 2, 2);
        }
        for (const h of cfg.hazards) {
          const x = m.toX(h[0]), y = m.toY(h[3]), w = (h[2]-h[0])/cfg.roomSize*m.bw, hh = (h[3]-h[1])/cfg.roomSize*m.bh;
          ctx.save();
          ctx.setLineDash([8, 7]);
          rr(x, y, w, hh, 14, 'rgba(127,29,29,.08)', 'rgba(248,113,113,.40)');
          ctx.restore();
          writeLabel('METEOR LANE', x+w/2, y+12, 8, 'rgba(254,202,202,.82)');
        }
        for (const [index, meteor] of cont.meteors.entries()) drawFlamingMeteor(meteor, m, index);
        ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 3; ctx.beginPath();
        cont.path.forEach((p, i) => { const x = m.toX(p[0]), y = m.toY(p[1]); if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
        ctx.stroke();
        const gx = m.toX(cfg.goal[0]), gy = m.toY(cfg.goal[1]);
        ctx.beginPath(); ctx.arc(gx, gy, cfg.goalRadius/cfg.roomSize*m.bw*1.5, 0, Math.PI*2);
        ctx.fillStyle = 'rgba(34,197,94,.28)'; ctx.fill(); ctx.strokeStyle = '#86efac'; ctx.lineWidth = 4; ctx.stroke();
        ctx.strokeStyle = '#bbf7d0'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(gx-34,gy); ctx.lineTo(gx+34,gy); ctx.moveTo(gx,gy-34); ctx.lineTo(gx,gy+34); ctx.stroke();
        writeLabel('PAD', gx, gy+5, 13, '#dcfce7');
        drawDrone(m.toX(cont.x), m.toY(cont.y));
        const dist = Math.hypot(cfg.goal[0]-cont.x, cfg.goal[1]-cont.y);
        updateStats(cont.score, cont.steps, cont.won ? 'LANDED' : (dist < 2 ? 'FINAL' : 'DESCENT'), cont.message + ' Distance: ' + dist.toFixed(2) + 'm');
      }
      function drawMovingBlocks() {
        drawArtBase(.22);
        ctx.fillStyle = 'rgba(2,6,23,.68)';
        ctx.fillRect(0,0,W,48);
        ctx.fillStyle = '#f5f3ff';
        ctx.font = '900 22px Segoe UI';
        ctx.fillText('Portal Hazard Run', 42, 28);
        ctx.fillStyle = 'rgba(221,214,254,.82)';
        ctx.font = '800 12px Segoe UI';
        ctx.fillText('moving 0.5m hazards / forward X-meter sensor', 270, 28);
        const bx = 55, by = 70, bw = 610, bh = 470;
        rr(bx, by, bw, bh, 22, 'rgba(22,10,42,.44)', '#a855f7');
        ctx.fillStyle = 'rgba(34,211,238,.10)';
        for (let i=0;i<9;i++) ctx.fillRect(bx+i*bw/9, by, 2, bh);
        const toX = x => bx + x/cfg.roomSize*bw;
        const toY = y => by + bh - y/cfg.roomSize*bh;
        for (const h of cfg.hazards) {
          const x = toX(h[0]), y = toY(h[3]), w = (h[2]-h[0])/cfg.roomSize*bw, hh = (h[3]-h[1])/cfg.roomSize*bh;
          rr(x, y, w, hh, 8, 'rgba(220,38,38,.42)', '#fca5a5');
          writeLabel('HOT', x+w/2, y+hh/2+4, 11, '#fee2e2');
        }
        for (const o of cont.obstacles) {
          const x = toX(o.x), y = toY(o.y);
          const radius = Math.max(13, cfg.obstacleWidth / cfg.roomSize * bw / 2);
          const pulse = radius + 6 + Math.sin(animTick() * .10 + o.x) * 4;
          const grad = ctx.createRadialGradient(x, y, 3, x, y, pulse + 10);
          grad.addColorStop(0, 'rgba(34,211,238,.96)');
          grad.addColorStop(.5, 'rgba(168,85,247,.58)');
          grad.addColorStop(1, 'rgba(168,85,247,0)');
          ctx.beginPath(); ctx.arc(x, y, pulse + 10, 0, Math.PI*2); ctx.fillStyle = grad; ctx.fill();
          ctx.beginPath(); ctx.arc(x, y, pulse, 0, Math.PI*2); ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 4; ctx.stroke();
        }
        const gx = toX(cfg.goal[0]), gy = toY(cfg.goal[1]);
        rr(gx-32, gy-32, 64, 64, 12, 'rgba(34,197,94,.28)', '#86efac');
        writeLabel('EXIT', gx, gy+5, 13, '#dcfce7');
        ctx.strokeStyle = '#fb923c'; ctx.lineWidth = 3; ctx.beginPath();
        cont.path.forEach((p, i) => { const x = toX(p[0]), y = toY(p[1]); if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
        ctx.stroke();
        const px = toX(cont.x), py = toY(cont.y);
        const heading = Math.atan2(-cont.vy, cont.vx || 0.0001);
        ctx.beginPath(); ctx.moveTo(px, py);
        ctx.arc(px, py, cfg.observationRange/cfg.roomSize*bw, heading - .55, heading + .55);
        ctx.closePath(); ctx.fillStyle = 'rgba(56,189,248,.08)'; ctx.fill();
        drawPortalScout(px, py);
        const dist = Math.hypot(cfg.goal[0]-cont.x, cfg.goal[1]-cont.y);
        const hx = cont.vx || (cfg.goal[0] - cont.x), hy = cont.vy || (cfg.goal[1] - cont.y);
        const visible = cont.obstacles.some(o => {
          const dx=o.x-cont.x, dy=o.y-cont.y;
          return Math.hypot(dx,dy) <= cfg.observationRange && dx*hx + dy*hy > 0;
        });
        updateStats(cont.score, cont.steps, cont.won ? 'ESCAPED' : (visible ? 'PORTAL AHEAD' : 'SEARCHING'), cont.message + ' Distance: ' + dist.toFixed(2) + 'm');
      }
      function drawTeleportRush() {
        drawArtBase(.20);
        ctx.fillStyle = 'rgba(2,6,23,.66)';
        ctx.fillRect(0,0,W,48);
        ctx.fillStyle = '#f5f3ff';
        ctx.font = '900 22px Segoe UI';
        ctx.fillText('Portal Teleport Run', 42, 28);
        ctx.fillStyle = 'rgba(221,214,254,.82)';
        ctx.font = '800 12px Segoe UI';
        ctx.fillText('portal gates / random exits / local sensor', 285, 28);
        const bx = 55, by = 70, bw = 610, bh = 470;
        rr(bx, by, bw, bh, 22, 'rgba(22,10,42,.42)', '#a855f7');
        for (let i = 0; i < 70; i++) {
          const x = bx + ((i * 83 + animTick() * 0.35) % bw);
          const y = by + ((i * 47) % bh);
          ctx.fillStyle = i % 6 === 0 ? 'rgba(34,211,238,.42)' : 'rgba(221,214,254,.18)';
          ctx.fillRect(x, y, 2, 2);
        }
        const toX = x => bx + x/cfg.roomSize*bw;
        const toY = y => by + bh - y/cfg.roomSize*bh;
        for (const h of cfg.hazards) {
          const x = toX(h[0]), y = toY(h[3]), w = (h[2]-h[0])/cfg.roomSize*bw, hh = (h[3]-h[1])/cfg.roomSize*bh;
          rr(x, y, w, hh, 10, 'rgba(220,38,38,.32)', '#fca5a5');
          writeLabel('VOID', x+w/2, y+hh/2+4, 11, '#fee2e2');
        }
        ctx.strokeStyle = '#c084fc'; ctx.lineWidth = 3; ctx.beginPath();
        cont.path.forEach((p, i) => { const x = toX(p[0]), y = toY(p[1]); if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y); });
        ctx.stroke();
        for (const [i, t] of cont.teleports.entries()) {
          const x = toX(t.x), y = toY(t.y);
          const pulse = 18 + Math.sin(animTick() * 0.11 + t.pulse) * 5;
          const grad = ctx.createRadialGradient(x, y, 4, x, y, pulse + 10);
          grad.addColorStop(0, 'rgba(34,211,238,.95)');
          grad.addColorStop(.5, 'rgba(168,85,247,.55)');
          grad.addColorStop(1, 'rgba(168,85,247,0)');
          ctx.beginPath(); ctx.arc(x, y, pulse + 10, 0, Math.PI*2); ctx.fillStyle = grad; ctx.fill();
          ctx.beginPath(); ctx.arc(x, y, pulse, 0, Math.PI*2); ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 4; ctx.stroke();
          ctx.beginPath(); ctx.arc(x, y, pulse * .55, 0, Math.PI*2); ctx.strokeStyle = '#f0abfc'; ctx.lineWidth = 3; ctx.stroke();
          writeLabel('T' + (i+1), x, y+4, 10, '#fdf4ff');
        }
        const gx = toX(cfg.goal[0]), gy = toY(cfg.goal[1]);
        rr(gx-34, gy-34, 68, 68, 16, 'rgba(34,197,94,.30)', '#86efac');
        writeLabel('EXIT', gx, gy+5, 13, '#dcfce7');
        const px = toX(cont.x), py = toY(cont.y);
        ctx.beginPath();
        ctx.arc(px, py, cfg.observationRange/cfg.roomSize*bw, 0, Math.PI*2);
        ctx.fillStyle = 'rgba(34,211,238,.07)';
        ctx.fill();
        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(animTick() * .045);
        ctx.strokeStyle = 'rgba(192,132,252,.72)';
        ctx.lineWidth = 3;
        ctx.beginPath(); ctx.arc(0, 0, 35, 0, Math.PI * 1.45); ctx.stroke();
        ctx.fillStyle = '#22d3ee';
        ctx.beginPath(); ctx.arc(35, 0, 4, 0, Math.PI*2); ctx.fill();
        ctx.restore();
        drawPortalScout(px, py);
        const dist = Math.hypot(cfg.goal[0]-cont.x, cfg.goal[1]-cont.y);
        const visible = cont.teleports.some(t => Math.hypot(t.x-cont.x, t.y-cont.y) <= cfg.observationRange);
        updateStats(cont.score, cont.steps, cont.won ? 'ESCAPED' : (visible ? 'PORTAL NEAR' : 'SEARCHING'), cont.message + ' Distance: ' + dist.toFixed(2) + 'm');
      }
      function drawContinuous() {
        if (cfg.teleportMode) drawTeleportRush();
        else if (cfg.obstacleMode) drawMovingBlocks();
        else drawDroneLanding();
      }
      function draw() {
        if (cfg.mode === 'grid') drawGrid();
        else drawContinuous();
        drawParticles();
        if (!cfg.replay && !resultShown) {
          const won = cfg.mode === 'grid' ? Boolean(grid && grid.won) : Boolean(cont && cont.won);
          if (won) {
            resultShown = true;
            gamePaused = true;
            showGameOverlay('MISSION COMPLETE', cfg.goalLabel + ' REACHED', 'Excellent run. Your score improves when you finish in fewer moves.', 'PLAY AGAIN');
          }
        }
      }
      let replayIndex = 0;
      let replayPlaying = false;
      let replayDelay = 450;
      let replayLastTick = 0;
      let replayDecorLast = 0;
      function applyReplayFrame(index) {
        if (!cfg.replay || !cfg.replay.states.length) return;
        replayIndex = clamp(index, 0, cfg.replay.states.length - 1);
        const state = cfg.replay.states[replayIndex];
        const score = cfg.replay.scores[replayIndex] || 0;
        const atEnd = replayIndex === cfg.replay.states.length - 1;
        if (cfg.mode === 'grid') {
          grid.pos = [state[0], state[1]];
          grid.visualFrom = grid.pos.slice();
          grid.visualTo = grid.pos.slice();
          grid.moveStarted = performance.now();
          grid.phase = cfg.kind === 'sarsa' ? 0 : (state[3] || 0);
          grid.score = score;
          grid.steps = replayIndex;
          grid.won = atEnd && cfg.replay.success;
          grid.collected = new Set();
          const mask = cfg.kind === 'sarsa' ? 0 : (state[2] || 0);
          for (let i = 0; i < grid.keys.length; i++) if (mask & (1 << i)) grid.collected.add(i);
          grid.visited = new Set(cfg.replay.states.slice(0, replayIndex + 1).map(s => s[0] + ',' + s[1]));
          if (cfg.kind === 'sarsa') grid.boxes = [[state[2], state[3]]];
          const action = cfg.replay.actions[replayIndex];
          if (action >= 0 && action <= 3) grid.dir = action;
          grid.message = atEnd
            ? (cfg.replay.success ? 'Episode finished successfully.' : 'Episode ended before the goal.')
            : 'Agent action ' + (action < 0 ? 'START' : ['UP','RIGHT','DOWN','LEFT'][action]) + '.';
        } else {
          cont.x = state[0]; cont.y = state[1];
          cont.vx = state[2] || 0; cont.vy = state[3] || 0;
          if (cfg.replay.obstacles && cfg.replay.obstacles[replayIndex]) {
            cont.obstacles = cfg.replay.obstacles[replayIndex].map(item => ({x:item[0], y:item[1], axis:item[2], direction:item[3], dx:item[2]===0?item[3]:0, dy:item[2]===1?item[3]:0}));
          }
          cont.score = score;
          cont.steps = replayIndex;
          cont.won = atEnd && cfg.replay.success;
          cont.path = cfg.replay.states.slice(0, replayIndex + 1).map(s => [s[0], s[1]]);
          cont.message = atEnd
            ? (cfg.replay.success ? 'Episode finished successfully.' : 'Episode ended before the goal.')
            : 'Replaying the agent path.';
        }
        const timeline = root.querySelector('#replayTimeline');
        if (timeline) timeline.value = replayIndex;
        const toggle = root.querySelector('#replayToggle');
        if (toggle) toggle.textContent = replayPlaying ? 'PAUSE' : 'PLAY';
      }
      function stepReplay(now) {
        if (!cfg.replay || !replayPlaying) return;
        if (!replayLastTick) replayLastTick = now;
        if (now - replayLastTick < replayDelay) return;
        replayLastTick = now;
        if (replayIndex >= cfg.replay.states.length - 1) {
          replayPlaying = false;
          applyReplayFrame(replayIndex);
          return;
        }
        applyReplayFrame(replayIndex + 1);
      }
      function updateReplayDecor(now) {
        if (!cont) return;
        cont.anim = now;
        if (!replayDecorLast) replayDecorLast = now;
        const scale = Math.min(2, (now - replayDecorLast) / 16.67);
        replayDecorLast = now;
        updateLanderMeteors(scale);
      }
      function loop(now) {
        if (cfg.replay) {
          stepReplay(now);
          if (cfg.mode === 'grid') updateGridWorld(now);
          else updateReplayDecor(now);
        } else if (!gameStarted || gamePaused) {
          // Keep rendering the live scene while the start or pause overlay is open.
        } else if (cfg.mode === 'continuous') {
          cont.anim = now;
          stepContinuous();
        }
        else updateGridWorld(now);
        draw();
        requestAnimationFrame(loop);
      }
      root.querySelector('#resetBtn').addEventListener('click', () => {
        cfg.mode === 'grid' ? initGrid() : initContinuous();
        resultShown = false;
        gamePaused = false;
        if (cfg.replay) {
          replayIndex = 0; replayPlaying = true; replayLastTick = 0; applyReplayFrame(0);
        } else {
          gameStarted = true;
          hideGameOverlay();
          pauseBtn.textContent = 'PAUSE';
          focusGame();
        }
        draw();
      });
      overlayAction.addEventListener('click', () => {
        if (resultShown) {
          cfg.mode === 'grid' ? initGrid() : initContinuous();
          resultShown = false;
        }
        gameStarted = true;
        gamePaused = false;
        pauseBtn.textContent = 'PAUSE';
        hideGameOverlay();
        focusGame();
      });
      pauseBtn.addEventListener('click', () => {
        if (cfg.replay || !gameStarted) return;
        gamePaused = !gamePaused;
        pauseBtn.textContent = gamePaused ? 'RESUME' : 'PAUSE';
        if (gamePaused) showGameOverlay('GAME PAUSED', cfg.title, 'Your run is frozen. Resume when you are ready.', 'RESUME');
        else { hideGameOverlay(); focusGame(); }
      });
      root.querySelectorAll('[data-grid]').forEach(btn => btn.addEventListener('click', () => moveGrid(Number(btn.dataset.grid))));
      root.querySelectorAll('[data-v]').forEach(btn => {
        const [vx,vy] = btn.dataset.v.split(',').map(Number);
        btn.addEventListener('mousedown', () => setVector(vx,vy));
        btn.addEventListener('touchstart', e => { e.preventDefault(); setVector(vx,vy); });
        btn.addEventListener('mouseup', () => setVector(0,0));
        btn.addEventListener('mouseleave', () => setVector(0,0));
        btn.addEventListener('touchend', () => setVector(0,0));
        btn.addEventListener('click', () => { setVector(vx,vy); setTimeout(() => setVector(0,0), 180); });
      });
      if (cfg.replay) {
        const timeline = root.querySelector('#replayTimeline');
        timeline.max = Math.max(0, cfg.replay.states.length - 1);
        timeline.addEventListener('input', () => {
          replayPlaying = false;
          applyReplayFrame(Number(timeline.value));
        });
        root.querySelector('#replayToggle').addEventListener('click', () => {
          if (replayIndex >= cfg.replay.states.length - 1) replayIndex = 0;
          replayPlaying = !replayPlaying;
          replayLastTick = 0;
          applyReplayFrame(replayIndex);
        });
        root.querySelector('#replayPrev').addEventListener('click', () => {
          replayPlaying = false;
          applyReplayFrame(replayIndex - 1);
        });
        root.querySelector('#replayNext').addEventListener('click', () => {
          replayPlaying = false;
          applyReplayFrame(replayIndex + 1);
        });
        root.querySelectorAll('[data-speed]').forEach(btn => btn.addEventListener('click', () => {
          replayDelay = Number(btn.dataset.speed);
          root.querySelectorAll('[data-speed]').forEach(item => item.classList.remove('active'));
          btn.classList.add('active');
          replayPlaying = true;
          replayLastTick = 0;
          applyReplayFrame(replayIndex);
        }));
      }
      const directionForKey = key => ({
        arrowup: 'up', w: 'up',
        arrowright: 'right', d: 'right',
        arrowdown: 'down', s: 'down',
        arrowleft: 'left', a: 'left'
      })[key];
      const actionForDirection = {up: 0, right: 1, down: 2, left: 3};
      root.addEventListener('keydown', e => {
        if (cfg.replay) return;
        const k = e.key.toLowerCase();
        if (k === 'escape') {
          pauseBtn.click();
          e.preventDefault();
          return;
        }
        const direction = directionForKey(k);
        if (!direction) return;
        e.preventDefault();
        setKeyboardActive(true);
        setKeycap(direction, true);
        if (!gameStarted) overlayAction.click();
        if (gamePaused) return;
        if (cfg.mode === 'grid') {
          if (!e.repeat) moveGrid(actionForDirection[direction]);
        } else {
          input[direction] = true;
        }
      });
      root.addEventListener('keyup', e => {
        const direction = directionForKey(e.key.toLowerCase());
        if (!direction) return;
        input[direction] = false;
        setKeycap(direction, false);
        e.preventDefault();
      });
      root.addEventListener('focusin', () => setKeyboardActive(!cfg.replay));
      root.addEventListener('focusout', e => {
        if (!root.contains(e.relatedTarget)) setKeyboardActive(false);
      });
      window.addEventListener('blur', () => {
        input.left = input.right = input.up = input.down = false;
        ['up', 'right', 'down', 'left'].forEach(direction => setKeycap(direction, false));
        setKeyboardActive(false);
      });
      canvas.addEventListener('pointerdown', focusGame);
      if (cfg.mode === 'grid') initGrid(); else initContinuous();
      if (cfg.replay) applyReplayFrame(0);
      loop();
      </script>
    </div>
    """
    components.html(template.replace("__CONFIG__", config), height=880, scrolling=False)


def init_manual(room_kind: str) -> None:
    key = f"manual_{room_kind}"
    if key in st.session_state:
        return
    env = make_env(room_kind, seed=31, slip=0.18)
    state = env.reset()
    st.session_state[key] = {
        "env": env,
        "trajectory": [{"state": state.copy() if isinstance(state, np.ndarray) else state, "action": None, "reward": 0.0, "done": False, "obstacles": getattr(env, "obstacles", [])}],
    }


def reset_manual(room_kind: str) -> None:
    st.session_state.pop(f"manual_{room_kind}", None)
    init_manual(room_kind)


def manual_step(room_kind: str, action: int) -> None:
    init_manual(room_kind)
    state = st.session_state[f"manual_{room_kind}"]
    env = state["env"]
    if state["trajectory"][-1].get("done"):
        return
    next_state, reward, done, info = env.step(action)
    state["trajectory"].append(
        {
            "state": next_state.copy() if isinstance(next_state, np.ndarray) else next_state,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info,
            "obstacles": [dict(item) for item in getattr(env, "obstacles", [])],
        }
    )


def render_play_tab(room_kind: str) -> None:
    arcade_component(room_kind)


def run_training(room_kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    seed = int(params["seed"])
    gamma = params["gamma"]
    if room_kind == "dp":
        env = GridEscapeRoom(room1_config(slip_probability=params["slip"], seed=seed))
        result = value_iteration(env, gamma=gamma, theta=params["theta"], max_iterations=params["max_iterations"])
    elif room_kind == "sarsa":
        env = SokobanEscapeRoom(room2_config(slip_probability=params["slip"], seed=seed))
        result = train_sarsa(env, episodes=params["episodes"], max_steps=params["max_steps"], alpha=params["alpha"], gamma=gamma, epsilon=params["epsilon"], epsilon_min=params["epsilon_min"], epsilon_decay=params["epsilon_decay"], seed=seed)
    elif room_kind == "q_learning":
        env = GridEscapeRoom(room3_config(slip_probability=params["slip"], seed=seed))
        result = train_q_learning(env, episodes=params["episodes"], max_steps=params["max_steps"], alpha=params["alpha"], gamma=gamma, epsilon=params["epsilon"], epsilon_min=params["epsilon_min"], epsilon_decay=params["epsilon_decay"], seed=seed)
    elif room_kind == "approx":
        env = ContinuousEscapeRoom(continuous_room_config(seed=seed))
        result = train_approx_q_learning(env, episodes=params["episodes"], max_steps=params["max_steps"], alpha=params["alpha"], gamma=gamma, epsilon=params["epsilon"], epsilon_min=params["epsilon_min"], epsilon_decay=params["epsilon_decay"], seed=seed)
    else:
        env = DynamicObstacleRoom(obstacle_room_config(seed=seed, obstacle_count=params["obstacle_count"], observation_range=params["observation_range"]))
        result = train_approx_q_learning(env, episodes=params["episodes"], max_steps=params["max_steps"], alpha=params["alpha"], gamma=gamma, epsilon=params["epsilon"], epsilon_min=params["epsilon_min"], epsilon_decay=params["epsilon_decay"], seed=seed)
    return {"room_kind": room_kind, "env": env, "result": result, "params": params}


def optimize_hyperparameters(room_kind: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if room_kind == "dp":
        variants = [
            {"gamma": 0.92, "theta": 1e-3},
            {"gamma": 0.96, "theta": 1e-4},
            {"gamma": 0.98, "theta": 1e-4},
            {"gamma": 0.99, "theta": 1e-5},
        ]
    else:
        variants = [
            {"alpha": params["alpha"], "gamma": params["gamma"], "epsilon": params["epsilon"], "epsilon_decay": params["epsilon_decay"]},
            {"alpha": max(0.01, params["alpha"] * 0.7), "gamma": 0.98, "epsilon": 0.30, "epsilon_decay": 0.995},
            {"alpha": min(0.5, params["alpha"] * 1.3), "gamma": 0.96, "epsilon": 0.50, "epsilon_decay": 0.990},
            {"alpha": params["alpha"], "gamma": 0.99, "epsilon": 0.35, "epsilon_decay": 0.993},
        ]

    rows: List[Dict[str, Any]] = []
    best_score = float("-inf")
    best_params = dict(params)
    for index, variant in enumerate(variants, start=1):
        candidate = dict(params)
        candidate.update(variant)
        candidate["seed"] = int(params["seed"]) + index * 101
        if room_kind != "dp":
            cap = 300 if room_kind in {"sarsa", "q_learning"} else 180
            candidate["episodes"] = min(int(params["episodes"]), cap)
        trial = run_training(room_kind, candidate)
        attempts = trial["result"].get("attempts", [])
        sample = attempts if room_kind == "dp" else attempts[-min(30, len(attempts)) :]
        success_rate = sum(bool(item["success"]) for item in sample) / max(1, len(sample))
        mean_reward = float(np.mean([float(item["reward"]) for item in sample])) if sample else float("-inf")
        mean_steps = float(np.mean([int(item["steps"]) for item in sample])) if sample else float("inf")
        score = success_rate * 10000.0 + mean_reward - mean_steps * 0.05
        row = {
            "candidate": index,
            "alpha": candidate.get("alpha", 0.0),
            "gamma": candidate["gamma"],
            "epsilon": candidate.get("epsilon", 0.0),
            "epsilon_decay": candidate.get("epsilon_decay", 0.0),
            "theta": candidate.get("theta", 0.0),
            "trial_episodes": candidate.get("episodes", 0),
            "success_rate": success_rate,
            "mean_reward": mean_reward,
            "mean_steps": mean_steps,
            "score": score,
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_params.update(variant)

    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    output_dir = Path("runs") / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{room_kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {"table": table, "best_params": best_params, "path": str(output_path)}


def register_training_run(run: Dict[str, Any]) -> float:
    run["artifacts"] = save_run_artifacts(run)
    st.session_state["last_run"] = run
    st.session_state.setdefault("runs_by_room", {})[run["room_kind"]] = run
    success_rate = run_success_rate(run)
    if success_rate >= 0.60:
        mark_room_completed(run["room_kind"])
    return success_rate


def render_train_guide(room_kind: str) -> None:
    is_dp = room_kind == "dp"
    strategy = (
        "<b>Convergence</b><span>Gamma sets the reward horizon; the stop threshold controls Value Iteration precision.</span>"
        if is_dp
        else "<b>Learning policy</b><span>Alpha controls update strength; epsilon controls exploration and decays over episodes.</span>"
    )
    environment = (
        "<b>Grid dynamics</b><span>Slip probability and the episode limit control movement uncertainty and search depth.</span>"
        if room_kind in {"dp", "sarsa", "q_learning"}
        else "<b>Continuous dynamics</b><span>The episode limit controls how long the agent can search the 10x10 meter room.</span>"
    )
    if room_kind == "obstacles":
        environment = "<b>Portal observation</b><span>Control the moving 0.5m hazards and how far ahead the agent can observe.</span>"

    st.markdown(
        f"""
        <div class="train-guide">
          <h3>Run setup</h3>
          <p>Set the environment and learning parameters, then launch a standard run or compare candidate configurations.</p>
          <div class="param-grid">
            <div class="param-card"><b>Reproducibility</b><span>Keep the seed fixed when comparing parameter changes.</span></div>
            <div class="param-card">{strategy}</div>
            <div class="param-card">{environment}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_train_tab(room_kind: str) -> None:
    section_header(
        "Training Console",
        "Configure agent",
        f"{ROOM_THEMES[room_kind]['algorithm']} parameters and environment controls.",
    )
    render_train_guide(room_kind)
    with st.form(f"train_form_{room_kind}"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="form-section">General Setup</div>', unsafe_allow_html=True)
            seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=100000,
                value=7,
                step=1,
                help="Controls randomness. Use the same seed to reproduce the same training behavior.",
            )
            gamma_default = 0.985 if room_kind in {"approx", "obstacles"} else 0.96
            gamma = st.slider(
                "Gamma - future reward importance",
                0.80,
                0.999,
                gamma_default,
                0.001,
                help="Discount factor. Higher gamma means the agent values long-term rewards more strongly.",
            )
        with col2:
            st.markdown('<div class="form-section">Environment</div>', unsafe_allow_html=True)
            if room_kind in {"dp", "sarsa", "q_learning"}:
                slip = st.slider(
                    "Slip probability - movement noise",
                    0.0,
                    0.6,
                    0.25 if room_kind == "dp" else 0.18,
                    0.01,
                    help="Chance that movement is redirected on slippery tiles.",
                )
            else:
                slip = 0.0
            max_steps = st.slider(
                "Max steps per episode",
                80 if room_kind in {"dp", "sarsa", "q_learning"} else 200,
                1600,
                250 if room_kind in {"dp", "sarsa"} else (1400 if room_kind == "obstacles" else 850),
                10 if room_kind in {"dp", "sarsa", "q_learning"} else 50,
                help="Maximum number of actions before an episode is stopped.",
            )
        with col3:
            st.markdown('<div class="form-section">Algorithm</div>', unsafe_allow_html=True)
            if room_kind == "dp":
                theta = st.select_slider(
                    "Stop threshold - DP convergence",
                    options=[1e-2, 1e-3, 1e-4, 1e-5],
                    value=1e-4,
                    help="Value Iteration stops when value changes are smaller than this threshold.",
                )
                max_iterations = st.slider(
                    "Max value-iteration sweeps",
                    50,
                    2000,
                    1000,
                    50,
                    help="Maximum number of full passes over all states.",
                )
                episodes = alpha = epsilon = epsilon_min = epsilon_decay = 0
            else:
                episodes = st.slider(
                    "Training episodes",
                    50,
                    2500,
                    650 if room_kind in {"sarsa", "q_learning"} else 450,
                    50,
                    help="Number of complete attempts used for learning.",
                )
                alpha = st.slider(
                    "Alpha - learning rate",
                    0.01,
                    0.5,
                    0.15 if room_kind in {"sarsa", "q_learning"} else 0.08,
                    0.01,
                    help="How strongly new experience updates the current value estimate.",
                )
                epsilon = st.slider(
                    "Initial epsilon - exploration",
                    0.0,
                    1.0,
                    0.4,
                    0.01,
                    help="Probability of choosing a random exploratory action at the beginning.",
                )
                epsilon_min = st.slider(
                    "Minimum epsilon",
                    0.0,
                    0.3,
                    0.03,
                    0.01,
                    help="Lowest exploration probability after decay.",
                )
                epsilon_decay = st.slider(
                    "Epsilon decay",
                    0.95,
                    1.0,
                    0.993,
                    0.001,
                    help="How quickly exploration decreases after each episode.",
                )
                theta = 1e-4
                max_iterations = 1000

        if room_kind == "obstacles":
            st.markdown('<div class="form-section">Dynamic Portal-Hazard Settings</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                obstacle_count = st.slider(
                    "Moving portal hazard count",
                    2,
                    15,
                    7,
                    1,
                    help="Number of moving 0.5-meter obstacles generated in each room.",
                )
            with c2:
                observation_range = st.slider(
                    "Forward observation range (meters)",
                    1.0,
                    6.0,
                    3.0,
                    0.5,
                    help="How far ahead the agent can observe the nearest obstacle, measured center to center.",
                )
        else:
            obstacle_count = 7
            observation_range = 3.0

        train_button, tune_button = st.columns(2)
        with train_button:
            submitted = st.form_submit_button("Start training", type="primary", use_container_width=True)
        with tune_button:
            optimize_submitted = st.form_submit_button("Optimize and train", use_container_width=True)

    params = {
        "seed": int(seed),
        "gamma": gamma,
        "slip": slip,
        "max_steps": max_steps,
        "theta": theta,
        "max_iterations": max_iterations,
        "episodes": episodes,
        "alpha": alpha,
        "epsilon": epsilon,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "obstacle_count": obstacle_count,
        "observation_range": observation_range,
    }
    if optimize_submitted:
        with st.spinner("Comparing hyperparameter candidates, then training the best configuration..."):
            tuning = optimize_hyperparameters(room_kind, params)
            st.session_state.setdefault("tuning_by_room", {})[room_kind] = tuning
            completed_run = run_training(room_kind, tuning["best_params"])
            success_rate = register_training_run(completed_run)
        st.success(
            f"Optimization complete. Best configuration trained with {success_rate * 100:.1f}% recent success."
        )
    elif submitted:
        with st.spinner("Training the agent and building arcade replays..."):
            completed_run = run_training(room_kind, params)
            success_rate = register_training_run(completed_run)
        attempt_count = len(completed_run["result"].get("attempts", []))
        st.success(
            f"Training complete. All {attempt_count} attempts are available in Replay. "
            f"Recent success rate: {success_rate * 100:.1f}%."
        )

    tuning = st.session_state.get("tuning_by_room", {}).get(room_kind)
    if tuning:
        st.subheader("Hyperparameter Optimization Results")
        st.dataframe(
            tuning["table"].drop(columns=["score"]),
            use_container_width=True,
            hide_index=True,
        )
        best = tuning["best_params"]
        st.info(
            "Best configuration: "
            + ", ".join(
                f"{name}={value}"
                for name, value in best.items()
                if name in {"alpha", "gamma", "epsilon", "epsilon_decay", "theta"}
            )
            + f". Comparison saved to {tuning['path']}."
        )


def get_current_run(room_kind: str) -> Dict[str, Any] | None:
    room_run = st.session_state.get("runs_by_room", {}).get(room_kind)
    if room_run:
        return room_run
    run = st.session_state.get("last_run")
    if run and run["room_kind"] == room_kind:
        return run
    return None


def render_replay_tab(room_kind: str) -> None:
    section_header(
        "Episode Archive",
        "Replay training episodes",
        "Filter the complete run history and inspect any recorded episode in the arcade player.",
    )
    run = get_current_run(room_kind)
    if not run:
        empty_state(
            "R",
            "No recorded episodes",
            "Complete a training run in this room to populate the replay archive.",
        )
        return
    result = run["result"]
    attempts = result.get("attempts", [])
    if not attempts:
        st.warning("No training attempts were recorded for this run.")
        return

    if room_kind == "obstacles":
        st.subheader("Unseen Random-Room Test")
        random_seed = st.number_input(
            "Random test seed",
            min_value=0,
            max_value=100000,
            value=int(run["params"]["seed"]) + 10000,
            step=1,
            key="obstacle_generalization_seed",
        )
        if st.button("Generate random room and test learned policy", use_container_width=True):
            test_env = DynamicObstacleRoom(
                obstacle_room_config(
                    seed=int(random_seed),
                    obstacle_count=int(run["params"]["obstacle_count"]),
                    observation_range=float(run["params"]["observation_range"]),
                )
            )
            trajectory = run_continuous_policy(
                test_env,
                result["agent"],
                max_steps=int(run["params"]["max_steps"]),
                seed=int(random_seed),
            )
            rewards = np.asarray([float(item["reward"]) for item in trajectory], dtype=np.float32)
            obstacle_frames = [
                [
                    [float(obstacle["x"]), float(obstacle["y"]), float(obstacle["axis"]), float(obstacle["direction"])]
                    for obstacle in item.get("obstacles", [])
                ]
                for item in trajectory
            ]
            test_attempt = {
                "episode": int(random_seed),
                "label": f"Unseen random room {int(random_seed)}",
                "states": np.asarray([item["state"] for item in trajectory], dtype=np.float32),
                "actions": np.asarray(
                    [-1 if item["action"] is None else int(item["action"]) for item in trajectory],
                    dtype=np.int8,
                ),
                "rewards": rewards,
                "reward": float(np.sum(rewards, dtype=np.float64)),
                "steps": max(0, len(trajectory) - 1),
                "success": bool(trajectory and trajectory[-1]["done"]),
                "epsilon": 0.0,
                "obstacles": np.asarray(obstacle_frames, dtype=np.float32),
            }
            st.session_state["obstacle_generalization_attempt"] = test_attempt

        test_attempt = st.session_state.get("obstacle_generalization_attempt")
        if test_attempt:
            outcome = "SUCCESS" if test_attempt["success"] else "FAILED"
            st.info(
                f"Unseen room result: {outcome}, {test_attempt['steps']} steps, "
                f"reward {test_attempt['reward']:.1f}."
            )
            arcade_component(room_kind, test_attempt)
            st.divider()

    successes = sum(bool(attempt["success"]) for attempt in attempts)
    best_reward = max(float(attempt["reward"]) for attempt in attempts)
    hud(
        room_kind,
        [
            ("recorded attempts", str(len(attempts))),
            ("successful", str(successes)),
            ("failed", str(len(attempts) - successes)),
            ("success rate", f"{successes / len(attempts) * 100:.1f}%"),
            ("best score", f"{best_reward:.1f}"),
        ],
    )
    st.markdown(
        '<div class="small-note"><b>Complete episode library.</b> Every row is one full agent episode recorded during training. Filter the library, choose an episode in the player, and inspect its exact movement step by step.</div>',
        unsafe_allow_html=True,
    )

    filter_col, sort_col = st.columns([1, 1])
    with filter_col:
        outcome = st.radio(
            "Episode result",
            ["All episodes", "Successful only", "Failed only"],
            horizontal=True,
            key=f"replay_outcome_{room_kind}",
        )
    with sort_col:
        ordering = st.selectbox(
            "Order attempts",
            ["Episode: newest first", "Episode: oldest first", "Score: highest first"],
            key=f"replay_order_{room_kind}",
        )

    filtered = filter_replay_attempts(attempts, outcome, ordering)

    if not filtered:
        st.warning("No episodes match this filter.")
        return

    total_episodes = max(int(attempt["episode"]) for attempt in attempts)
    library = pd.DataFrame(replay_library_rows(filtered, total_episodes))
    st.subheader(f"Episode Library ({len(filtered)} shown of {len(attempts)})")
    st.dataframe(
        library,
        width="stretch",
        height=min(520, 42 + 35 * len(library)),
        hide_index=True,
        column_config={
            "Episode": st.column_config.NumberColumn("Episode", format="%d"),
            "Result": st.column_config.TextColumn("Result"),
            "Reward": st.column_config.NumberColumn("Total reward", format="%.2f"),
            "Steps": st.column_config.NumberColumn("Steps", format="%d"),
            "Epsilon": st.column_config.NumberColumn("Exploration epsilon", format="%.4f"),
            "Training phase": st.column_config.TextColumn("Training phase"),
        },
    )

    attempts_by_episode = {int(attempt["episode"]): attempt for attempt in filtered}
    episode_options = [int(attempt["episode"]) for attempt in filtered]
    selection_key = f"replay_selected_episode_{room_kind}"
    if st.session_state.get(selection_key) not in episode_options:
        st.session_state[selection_key] = episode_options[0]

    def episode_label(episode: int) -> str:
        attempt = attempts_by_episode[episode]
        outcome_label = "SUCCESS" if attempt["success"] else "FAILED"
        epsilon_label = f" | epsilon {float(attempt['epsilon']):.3f}" if "epsilon" in attempt else ""
        return (
            f"Episode {episode} | {outcome_label} | "
            f"score {float(attempt['reward']):.1f} | {int(attempt['steps'])} steps{epsilon_label}"
        )

    def move_episode(delta: int) -> None:
        current = int(st.session_state.get(selection_key, episode_options[0]))
        current_index = episode_options.index(current) if current in episode_options else 0
        next_index = min(len(episode_options) - 1, max(0, current_index + delta))
        st.session_state[selection_key] = episode_options[next_index]

    current_episode = int(st.session_state[selection_key])
    current_index = episode_options.index(current_episode)
    st.subheader("Replay Player")
    previous_col, select_col, next_col = st.columns([0.22, 0.56, 0.22])
    with previous_col:
        st.button(
            "Previous episode",
            key=f"replay_previous_{room_kind}",
            use_container_width=True,
            disabled=current_index == 0,
            on_click=move_episode,
            args=(-1,),
        )
    with select_col:
        selected_episode = st.selectbox(
            "Choose any episode to replay",
            episode_options,
            format_func=episode_label,
            key=selection_key,
        )
    with next_col:
        st.button(
            "Next episode",
            key=f"replay_next_{room_kind}",
            use_container_width=True,
            disabled=current_index == len(episode_options) - 1,
            on_click=move_episode,
            args=(1,),
        )

    selected_index = episode_options.index(int(selected_episode))
    selected_attempt = attempts_by_episode[int(selected_episode)]
    st.caption(
        f"Showing episode {selected_index + 1} of {len(episode_options)} in the current view. "
        "Use the controls inside the game to play, pause, step, scrub, and change speed."
    )
    arcade_component(room_kind, selected_attempt)


def save_metrics(df: pd.DataFrame, room_kind: str) -> str:
    os.makedirs("runs", exist_ok=True)
    filename = f"runs/{room_kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    return filename


def save_run_artifacts(run: Dict[str, Any]) -> Dict[str, str]:
    room_kind = run["room_kind"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path("runs") / f"{room_kind}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = metric_dataframe(run["result"]["metrics"])
    metrics_path = run_dir / "metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    attempts = run["result"].get("attempts", [])
    attempt_summary = pd.DataFrame(
        [
            {
                "episode": int(item["episode"]),
                "reward": float(item["reward"]),
                "steps": int(item["steps"]),
                "success": bool(item["success"]),
                "epsilon": float(item.get("epsilon", 0.0)),
            }
            for item in attempts
        ]
    )
    attempts_path = run_dir / "attempts.csv"
    attempt_summary.to_csv(attempts_path, index=False, encoding="utf-8-sig")

    params_path = run_dir / "parameters.json"
    params_path.write_text(json.dumps(run["params"], indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    x_name = "iteration" if room_kind == "dp" else "episode"
    x = metrics[x_name]
    if room_kind == "dp":
        axes[0, 0].plot(x, metrics["delta"], color="#2563eb")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_title("Convergence delta")
        axes[0, 1].plot(x, metrics["value_start"], color="#059669")
        axes[0, 1].set_title("Start-state value")
        rollout_steps = [int(item["steps"]) for item in attempts]
        rollout_rewards = [float(item["reward"]) for item in attempts]
        axes[1, 0].bar(range(1, len(rollout_steps) + 1), rollout_steps, color="#7c3aed")
        axes[1, 0].set_title("Policy rollout steps")
        axes[1, 1].bar(range(1, len(rollout_rewards) + 1), rollout_rewards, color="#ea580c")
        axes[1, 1].set_title("Policy rollout reward")
    else:
        axes[0, 0].plot(x, metrics["reward"], alpha=0.28, color="#64748b")
        axes[0, 0].plot(x, metrics["reward_ma_25"], color="#2563eb")
        axes[0, 0].set_title("Reward and moving average")
        axes[0, 1].plot(x, metrics["steps"], color="#7c3aed")
        axes[0, 1].set_title("Steps per episode")
        axes[1, 0].plot(x, metrics["success_rate_ma_25"], color="#059669")
        axes[1, 0].set_ylim(-0.02, 1.02)
        axes[1, 0].set_title("Success rate (25 episodes)")
        axes[1, 1].plot(x, metrics["epsilon"], color="#ea580c")
        axes[1, 1].set_title("Exploration epsilon")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel(x_name.title())
    figure.suptitle(f"{ROOM_THEMES[room_kind]['title']} learning report")
    plot_path = run_dir / "learning_report.png"
    figure.savefig(plot_path, dpi=150, facecolor="white")
    plt.close(figure)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "room": room_kind,
                "success_rate": run_success_rate(run),
                "attempts": len(attempts),
                "best_reward": max((float(item["reward"]) for item in attempts), default=0.0),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "directory": str(run_dir),
        "metrics": str(metrics_path),
        "attempts": str(attempts_path),
        "parameters": str(params_path),
        "plot": str(plot_path),
        "summary": str(summary_path),
    }


def render_analytics_tab(room_kind: str) -> None:
    section_header(
        "Learning Report",
        "Training analytics",
        "Reward, success, exploration, convergence, and approximation quality for the latest run.",
    )
    run = get_current_run(room_kind)
    if not run:
        empty_state(
            "A",
            "No analytics available",
            "Complete a training run to generate metrics and learning charts.",
        )
        return
    df = metric_dataframe(run["result"]["metrics"])
    if room_kind == "dp":
        last = df.iloc[-1]
        hud(room_kind, [("iterations", str(int(last["iteration"]))), ("delta", f"{float(last['delta']):.5f}"), ("start value", f"{float(last['value_start']):.2f}"), ("algorithm", "value iteration"), ("room", ROOM_THEMES[room_kind]["title"])])
        st.line_chart(df.set_index("iteration")[["delta", "value_start"]])
    else:
        last = df.iloc[-1]
        success_rate = float(df.tail(min(50, len(df)))["success"].mean()) if "success" in df else 0.0
        hud(room_kind, [("episodes", str(int(last["episode"]))), ("last score", f"{float(last['reward']):.1f}"), ("steps", str(int(last["steps"]))), ("success", f"{success_rate * 100:.0f}%"), ("epsilon", f"{float(last.get('epsilon', 0.0)):.3f}")])
        indexed = df.set_index("episode")
        st.markdown("#### Learning performance")
        st.line_chart(indexed[["reward", "reward_ma_25"]])
        performance_columns = [col for col in ["steps", "success_rate_ma_25"] if col in indexed.columns]
        st.line_chart(indexed[performance_columns])
        st.markdown("#### Exploration schedule")
        st.line_chart(indexed[["epsilon"]])
        if "mean_abs_td_error" in indexed.columns:
            st.markdown("#### Approximation error")
            st.line_chart(indexed[["mean_abs_td_error"]])

    artifacts = run.get("artifacts", {})
    plot_path = artifacts.get("plot")
    if plot_path and Path(plot_path).exists():
        st.caption(f"Saved learning report: {plot_path}")
        st.image(plot_path, use_container_width=True)

    download_col, save_col = st.columns(2)
    with download_col:
        st.download_button(
            "Download metrics CSV",
            data=df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{room_kind}_metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with save_col:
        if st.button("Save metrics in runs folder", use_container_width=True):
            st.success(f"Saved: {save_metrics(df, room_kind)}")


def render_details_tab(room_kind: str) -> None:
    t = ROOM_THEMES[room_kind]
    env = make_env(room_kind)
    is_grid = isinstance(env, GridEscapeRoom)
    model = "Known transition model" if room_kind == "dp" else "Unknown transition model"
    representation = "10x10 tabular grid" if is_grid else "Continuous 10x10 meter room"
    actions = "4 discrete directions" if is_grid else "9 discrete velocity pairs"
    terminal = f"Single terminal: {t['goal']}"

    section_header(
        "Environment Specification",
        "Room design",
        "State, action, transition, terminal, and reward definitions used by the agent.",
    )
    detail_values = [
        ("Algorithm", t["algorithm"]),
        ("Model", model),
        ("State", t["state"]),
        ("Actions", actions),
    ]
    detail_cards = "".join(
        f'<div class="detail-card"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in detail_values
    )
    st.markdown(
        f'<div class="details-grid" style="{style_vars(room_kind)}">{detail_cards}</div>',
        unsafe_allow_html=True,
    )

    if is_grid:
        config_data = {
            "representation": representation,
            "start": env.config.start,
            "final_state": env.config.goal,
            "terminal_condition": terminal,
            "walls": sorted(env.config.walls),
            "slippery_cells": sorted(env.config.slippery),
            "slip_probability": env.config.slip_probability,
            "traps": {str(position): reward for position, reward in env.config.traps.items()},
            "bonuses": {str(position): reward for position, reward in env.config.bonuses.items()},
            "required_items": env.config.keys,
            "box_start": env.config.box_start,
            "box_target": env.config.box_target,
            "portals": {str(source): target for source, target in env.config.portals.items()},
            "guard_cycles": [[list(position) for position in cycle] for cycle in env.config.guard_cycles],
        }
        environment_rows = [
            ("Representation", representation),
            ("Start", str(env.config.start)),
            ("Terminal", str(env.config.goal)),
            ("Slippery cells", str(len(env.config.slippery))),
            ("Walls", str(len(env.config.walls))),
        ]
        reward_rows = [
            ("Every action", env.config.step_reward),
            ("Reach final state", env.config.goal_reward),
            ("Collect required item", env.config.key_reward),
            ("Hit moving guard", env.config.guard_reward),
            ("Use portal", env.config.portal_reward),
        ]
    else:
        config_data = {
            "representation": representation,
            "room_size": env.config.room_size,
            "start": env.config.start,
            "final_state": env.config.goal,
            "terminal_condition": terminal,
            "actions": CONTINUOUS_ACTIONS,
            "dt": env.config.dt,
            "speed": env.config.speed,
            "hazards": env.config.hazards,
            "obstacle_width": getattr(env.config, "obstacle_width", None),
            "obstacle_count": getattr(env.config, "obstacle_count", None),
            "observation_range": getattr(env.config, "observation_range", None),
        }
        environment_rows = [
            ("Representation", representation),
            ("Time step", f"{env.config.dt:.2f} seconds"),
            ("Velocity values", "Vx,Vy in {-1,0,1}"),
            ("Start", str(env.config.start)),
            ("Terminal", str(env.config.goal)),
        ]
        reward_rows = [
            ("Every time step", env.config.step_reward),
            ("Reach final state", env.config.goal_reward),
            ("Hit wall", env.config.wall_penalty),
            ("Enter hazard", env.config.hazard_penalty),
            ("Progress shaping", env.config.progress_scale),
        ]
        if room_kind == "obstacles":
            reward_rows.append(("Hit moving obstacle", env.config.obstacle_penalty))

    environment_markup = "".join(
        f'<div class="reward-row"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in environment_rows
    )
    rewards_markup = "".join(
        f'<div class="reward-row"><span>{html.escape(label)}</span><strong>{float(value):+.3f}</strong></div>'
        for label, value in reward_rows
    )
    environment_col, rewards_col = st.columns(2)
    with environment_col:
        st.markdown(
            f'<div class="spec-band"><h3>Environment</h3>{environment_markup}</div>',
            unsafe_allow_html=True,
        )
    with rewards_col:
        st.markdown(
            f'<div class="spec-band"><h3>Reward function</h3>{rewards_markup}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Full environment configuration"):
        st.json(config_data)


def choose_room(room_kind: str) -> None:
    st.session_state["selected_room"] = room_kind


def clear_room_selection() -> None:
    st.session_state.pop("selected_room", None)


def render_room_selection() -> None:
    completed = set(st.session_state.get("completed_rooms", []))
    completed_count = len(completed)
    progress_percent = completed_count / len(ROOM_ORDER) * 100
    st.markdown(
        f"""
        <div class="select-head">
          <div>
            <div class="select-eyebrow">Campaign Control</div>
            <h2>Select a training room</h2>
            <p>Choose an environment, enter the arena, and train the matching reinforcement-learning agent.</p>
          </div>
          <div class="campaign-score">
            <strong>{completed_count}/{len(ROOM_ORDER)}</strong>
            <span>Rooms cleared</span>
          </div>
        </div>
        <div class="campaign-track" style="--progress:{progress_percent:.1f}%"><span></span></div>
        """,
        unsafe_allow_html=True,
    )

    rows = [ROOM_ORDER[:3], ROOM_ORDER[3:]]
    for row in rows:
        columns = st.columns(len(row))
        for column, room_kind in zip(columns, row):
            t = ROOM_THEMES[room_kind]
            room_number = ROOM_ORDER.index(room_kind) + 1
            is_completed = room_kind in completed
            status = "COMPLETED" if is_completed else ("START HERE" if room_kind == "dp" else "READY")
            status_class = "completed" if is_completed else "ready"
            objectives = "".join(f"<span>{html.escape(item)}</span>" for item in t["objectives"])
            with column:
                st.markdown(
                    f"""
                    <div class="room-select-card" style="{style_vars(room_kind)}">
                      <div class="room-card-top">
                        <span class="room-number">Room {room_number:02d}</span>
                        <span class="room-status {status_class}">{status}</span>
                      </div>
                      <div class="screen preview-{room_kind}" style="background-image:linear-gradient(180deg,rgba(2,6,23,.03),rgba(2,6,23,.24)),url('app/static/game_art/{html.escape(t['thumbnail_art'])}');background-size:cover;background-position:center;">
                        {thumbnail_markup(room_kind)}
                      </div>
                      <h3>{html.escape(t["title"])}</h3>
                      <div class="classic">{html.escape(t["inspiration"])}</div>
                      <div class="algo">{html.escape(t["algorithm"])}</div>
                      <div class="room-state">STATE · {html.escape(t["state"])}</div>
                      <p>{html.escape(t["mission"])}</p>
                      <div class="room-objectives">{objectives}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button(f"Enter room {room_number:02d}", key=f"open_{room_kind}", use_container_width=True):
                    choose_room(room_kind)
                    st.rerun()


def render_room_nav(room_kind: str) -> None:
    t = ROOM_THEMES[room_kind]
    room_index = ROOM_ORDER.index(room_kind)
    previous_kind = ROOM_ORDER[room_index - 1] if room_index > 0 else None
    next_kind = ROOM_ORDER[room_index + 1] if room_index + 1 < len(ROOM_ORDER) else None
    with st.container(key="room_navigation"):
        all_rooms, previous, context, next_room = st.columns([0.16, 0.13, 0.55, 0.16], vertical_alignment="center")
        with all_rooms:
            if st.button("All rooms", use_container_width=True):
                clear_room_selection()
                st.rerun()
        with previous:
            if st.button("Previous", use_container_width=True, disabled=previous_kind is None):
                choose_room(previous_kind)
                st.rerun()
        with context:
            st.markdown(
                f"""
                <div class="room-nav" style="{style_vars(room_kind)}">
                  <div>
                    <div class="crumb">Campaign / Room {room_index + 1:02d}</div>
                    <div class="current">{html.escape(t["title"])}</div>
                  </div>
                  <div class="hint">{html.escape(t["algorithm"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with next_room:
            if st.button("Next room", use_container_width=True, disabled=next_kind is None):
                choose_room(next_kind)
                st.rerun()


def render_room_app(room_kind: str) -> None:
    render_room_nav(room_kind)
    room_intro(room_kind)
    tabs = st.tabs(["Play Game", "Train Agent", "Episode Replay", "Analytics", "Room Specs"])
    with tabs[0]:
        render_play_tab(room_kind)
    with tabs[1]:
        render_train_tab(room_kind)
    with tabs[2]:
        render_replay_tab(room_kind)
    with tabs[3]:
        render_analytics_tab(room_kind)
    with tabs[4]:
        render_details_tab(room_kind)

    completed = set(st.session_state.get("completed_rooms", []))
    next_kind = next_room_kind(room_kind)
    if room_kind in completed and next_kind:
        st.success(f"Agent solved this room. The next room is now ready: {ROOM_THEMES[next_kind]['title']}.")
        if st.button(f"Continue to {ROOM_THEMES[next_kind]['title']}", type="primary", use_container_width=True):
            choose_room(next_kind)
            st.rerun()
    elif room_kind not in completed:
        st.info("Train an agent to at least 60% recent success to complete this room and continue the campaign.")
    else:
        st.success("Campaign complete. All escape rooms have been solved.")


def main() -> None:
    css()
    room_kind = st.session_state.get("selected_room")
    header(room_kind)
    if not room_kind:
        render_room_selection()
        return
    render_room_app(room_kind)


if __name__ == "__main__":
    main()
