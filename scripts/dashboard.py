"""
Interactive Stone Tool Embedding Dashboard.

Layout
------
  Top-left   : PCA scatter  (click a point to inspect)
  Top-centre : Stone detail panel  (renders + metadata + similar stones)
                OR archetype vertex panel (3 closest stones) when a vertex is clicked
  Top-right  : 3D PCA with Pareto archetype tetrahedron
                (click a stone → highlight in 2D; click A1-A4 vertex → archetype panel)

Usage
-----
  python scripts/dashboard.py
  python scripts/dashboard.py --run_dir outputs/dino_triptych/20260624_123456
  python scripts/dashboard.py --renders_dir outputs/renders_v2 --port 8050
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import dash
from dash import dcc, html, Input, Output

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import get_logger
from scripts.demo_embed_pca import load_site_map, _PALETTE

log = get_logger("dashboard")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

OUTPUT_BASE = ROOT / "outputs" / "dino_triptych"
RENDERS_DIR = ROOT / "outputs" / "multiview_renders"
XLSX_SITE   = ROOT / "wrl" / "Handaxes 2026 list with sites.xlsx"
XLSX_META   = ROOT / "wrl" / "Handaxes2026list_with_sites_and_metadata.xlsx"

# 6-view multiview renders  (2 rows × 3 cols in the detail panel)
RENDER_SLOTS = [
    ("pZ", "Top (↓)"),
    ("nZ", "Bottom (↑)"),
    ("pX", "Right (←)"),
    ("nX", "Left (→)"),
    ("pY", "Front (←)"),
    ("nY", "Back (→)"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _latest_run(base: Path) -> Path | None:
    runs = sorted(
        (d for d in base.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime, reverse=True,
    )
    return runs[0] if runs else None


def _b64(path: Path) -> str | None:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


def _site_key(site_full: str) -> str:
    """'Emek Refaim (ERM_600513)' → 'Emek Refaim'"""
    return site_full.split("(")[0].strip().rstrip(" -_")


def _site_colors(stems: list[str], site_map: dict) -> tuple[list, list, dict]:
    stem_sites   = [site_map.get(s, "Unknown") for s in stems]
    unique_sites = sorted(set(stem_sites))
    color        = {site: _PALETTE[i % len(_PALETTE)] for i, site in enumerate(unique_sites)}
    return stem_sites, unique_sites, color


# ---------------------------------------------------------------------------
# Metadata loading
# ---------------------------------------------------------------------------

META_COLS = ["Date / Period", "Location", "Elevation", "References", "Link"]

def load_metadata(xlsx: Path) -> dict[str, dict]:
    """Return {site_key: {col_name: value, ...}} from the metadata Excel."""
    if not xlsx.exists():
        return {}
    try:
        df = pd.read_excel(xlsx, header=0)
        result: dict[str, dict] = {}
        for _, row in df.iterrows():
            site_val = row.iloc[0]
            if not site_val or str(site_val).strip() == "":
                continue
            if str(site_val).strip().startswith("Site Name"):
                continue
            key = _site_key(str(site_val))
            entry: dict[str, str] = {}
            for col_name, val in zip(META_COLS, row.iloc[1:]):
                if val and str(val).strip() and str(val) != "nan":
                    entry[col_name] = str(val).strip()
            result[key] = entry
        return result
    except Exception as e:
        log.warning(f"Could not load metadata: {e}")
        return {}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all(run_dir: Path, renders_dir: Path,
             site_xlsx: Path, meta_xlsx: Path):
    emb_files = sorted((run_dir / "embeddings").glob("*.npy"))
    if not emb_files:
        raise FileNotFoundError(f"No embeddings found in {run_dir / 'embeddings'}")

    stems = [f.stem for f in emb_files]
    E     = np.stack([np.load(f) for f in emb_files]).astype(np.float32)  # [N, D]

    pca     = PCA(n_components=min(10, len(stems) - 1))
    coords  = pca.fit_transform(E)                         # [N, ≤10]
    var     = pca.explained_variance_ratio_

    sim = (normalize(E) @ normalize(E).T).astype(np.float32)  # [N, N] cosine sim

    site_map, notes_map = {}, {}
    if site_xlsx.exists():
        try:
            site_map, notes_map = load_site_map(site_xlsx)
        except Exception as e:
            log.warning(f"site map: {e}")

    meta = load_metadata(meta_xlsx)

    # Archetype data saved by dino_triptych.py plot_pca_3d (optional)
    arch_Z = np.load(run_dir / "archetypes_Z.npy") if (run_dir / "archetypes_Z.npy").exists() else None
    arch_S = np.load(run_dir / "archetypes_S.npy") if (run_dir / "archetypes_S.npy").exists() else None

    return stems, E, coords, var, sim, site_map, notes_map, meta, arch_Z, arch_S


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _dark_layout(**kw) -> dict:
    # Shared Plotly figure defaults. Pass height=N or margin=dict(...) to override.
    # ↳ height: controls the pixel height of PCA figures
    # ↳ margin l/r/t/b: inner whitespace (pixels) around the plot area
    base = dict(
        paper_bgcolor="#0f0f1a", plot_bgcolor="#0f0f1a",
        font=dict(color="white", size=11),
        margin=dict(l=50, r=20, t=45, b=45),
    )
    base.update(kw)
    return base


def build_pca(stems, coords, var, site_map, notes_map,
              selected_idx: int | None = None) -> go.Figure:
    stem_sites, unique_sites, color = _site_colors(stems, site_map)

    fig = go.Figure()
    for site in unique_sites:
        idxs = [i for i, s in enumerate(stem_sites) if s == site]
        fig.add_trace(go.Scatter(
            x=coords[idxs, 0], y=coords[idxs, 1],
            mode="markers", name=site,
            marker=dict(
                size=9,              # ← PCA dot size (pixels)
                color=color[site],
                line=dict(width=1, color="white"),  # ← dot border
            ),
            # Suppress Plotly's default selection dimming so points look the
            # same whether or not something is selected (we use the yellow ring
            # for the selected stone instead).
            selected={"marker": {"opacity": 1, "size": 9}},
            unselected={"marker": {"opacity": 1}},
            text=[stems[i] for i in idxs],
            customdata=idxs,
            hovertemplate="<b>%{text}</b><br>" + site + "<extra></extra>",
        ))

    if selected_idx is not None:
        fig.add_trace(go.Scatter(
            x=[coords[selected_idx, 0]], y=[coords[selected_idx, 1]],
            mode="markers", name="selected", showlegend=False,
            marker=dict(
                size=18,                          # ← selection ring size
                color="rgba(0,0,0,0)",
                line=dict(width=2.5, color="yellow"),
            ),
            hoverinfo="skip",
        ))

    n_pc = len(var)
    # Compute axis ranges from ALL points once so they never change on click.
    # 5% padding on each side keeps points away from the edge.
    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.05 or 1
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.05 or 1
    x_range = [coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x]
    y_range = [coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y]

    fig.update_layout(
        **_dark_layout(height=640),  # ← PCA figure height (pixels)
        title=f"PCA — PC1 {var[0]:.1%} | PC2 {var[1]:.1%}",
        xaxis=dict(title=f"PC1 ({var[0]:.1%})", gridcolor="#222", color="#aaa",
                   range=x_range, fixedrange=True),   # ← locked; never changes on click
        yaxis=dict(title=f"PC2 ({var[1]:.1%})", gridcolor="#222", color="#aaa",
                   range=y_range, fixedrange=True),   # ← locked; never changes on click
        legend=dict(
            bgcolor="#1a1a2e", bordercolor="#333", font_size=9,
            itemclick="toggleothers",   # ← click legend item → show only that site
            itemdoubleclick=False,
        ),
        clickmode="event+select",
    )
    return fig


def build_pca3d(stems, coords, var, site_map, notes_map,
                arch_Z: np.ndarray | None = None,
                arch_S: np.ndarray | None = None) -> go.Figure:
    """Interactive 3D PCA with Pareto archetype tetrahedron.
    Archetype vertex markers (A1-A4) are clickable — detected via text in clickData."""
    stem_sites, unique_sites, color = _site_colors(stems, site_map)
    coords3 = coords[:, :3]

    fig = go.Figure()

    # ── stone scatter (one trace per site) ───────────────────────────────────
    for site in unique_sites:
        idxs = [i for i, s in enumerate(stem_sites) if s == site]
        fig.add_trace(go.Scatter3d(
            x=coords3[idxs, 0], y=coords3[idxs, 1], z=coords3[idxs, 2],
            mode="markers", name=site,
            marker=dict(size=4, color=color[site], opacity=0.75),
            text=[stems[i] for i in idxs],
            customdata=idxs,
            hovertemplate="<b>%{text}</b><br>" + site + "<extra></extra>",
        ))

    # ── tetrahedron ──────────────────────────────────────────────────────────
    if arch_Z is not None:
        x, y, z = arch_Z[:, 0], arch_Z[:, 1], arch_Z[:, 2]

        # Faint translucent faces
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0, 0, 0, 1], j=[1, 1, 2, 2], k=[2, 3, 3, 3],
            color="white", opacity=0.06,
            flatshading=True, showlegend=False, hoverinfo="skip",
            name="simplex",
        ))

        # Glow halos (not interactive — no customdata, text, or hovertemplate)
        for sz, op in [(32, 0.04), (18, 0.12)]:
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="markers",
                marker=dict(size=sz, color="#00e5ff", opacity=op),
                showlegend=False, hoverinfo="skip",
            ))

        # Clickable vertex markers — identified in clickData by text="A1".."A4"
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers+text",
            marker=dict(
                size=9, color="white", opacity=1.0,
                line=dict(width=4, color="#00e5ff"),
            ),
            text=["A1", "A2", "A3", "A4"],
            textfont=dict(color="#00e5ff", size=13),
            textposition="top center",
            name="Archetypes",
            customdata=[0, 1, 2, 3],   # archetype index k
            hovertemplate="<b>%{text}</b> — click to explore<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#0f0f1a",
        font=dict(color="white", size=10),
        margin=dict(l=0, r=0, t=40, b=0),
        height=640,  # ← 3D PCA figure height (pixels)
        title=dict(
            text=(f"3D PCA — {var[0]:.1%} | {var[1]:.1%} | {var[2]:.1%}"
                  if len(var) >= 3 else "3D PCA"),
            font=dict(size=12),
        ),
        scene=dict(
            bgcolor="#0a0a18",
            xaxis=dict(title="PC1", gridcolor="#1e1e3a", color="#888",
                       backgroundcolor="#0a0a18", showbackground=True,
                       showspikes=False),
            yaxis=dict(title="PC2", gridcolor="#1e1e3a", color="#888",
                       backgroundcolor="#0a0a18", showbackground=True,
                       showspikes=False),
            zaxis=dict(title="PC3", gridcolor="#1e1e3a", color="#888",
                       backgroundcolor="#0a0a18", showbackground=True,
                       showspikes=False),
        ),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#333", font_size=9),
        clickmode="event",
    )
    return fig


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------

def _render_img(b64: str | None, label: str) -> html.Div:
    # One render thumbnail inside the detail panel grid cell.
    # Image width is 100% of its flex cell — cell width = panel width / 3.
    # ↳ "imageRendering": "pixelated" keeps depth maps crisp (no blur smoothing)
    # ↳ fontSize on the label: tweak the caption text size
    if b64:
        child = html.Img(
            src=f"data:image/png;base64,{b64}",
            style={"width": "100%", "imageRendering": "pixelated",
                   "borderRadius": "3px", "display": "block"},
        )
    else:
        child = html.Div("—", style={"color": "#444", "textAlign": "center",
                                      "padding": "20px 0"})
    return html.Div([
        html.P(label, style={"fontSize": "9px", "color": "#888",  # ← caption font size
                              "margin": "0 0 2px 0", "textAlign": "center"}),
        child,
    ], style={"flex": "1 1 0"})  # ← each image takes equal share of the row width


def build_detail(stem: str, idx: int, stems: list[str], sim: np.ndarray,
                 site_map: dict, notes_map: dict, meta: dict,
                 color: dict, renders_dir: Path) -> list:
    site  = site_map.get(stem, "Unknown")
    note  = notes_map.get(stem, "")
    skey  = _site_key(site)
    smeta = meta.get(skey, {})

    # Render images — 2 rows × 3 cols (top/bottom/right in row 1, left/front/back in row 2)
    # ↳ To change layout: adjust the slice indices slots[:3] / slots[3:]
    # ↳ Gap between images: "gap" value (px) in each row's style
    slots = [(rname, label) for rname, label in RENDER_SLOTS]
    rows_imgs = []
    for row_slots in [slots[:3], slots[3:]]:
        row_imgs = [_render_img(_b64(renders_dir / f"{stem}_{rname}.png"), label)
                    for rname, label in row_slots]
        rows_imgs.append(html.Div(row_imgs, style={"display": "flex", "gap": "4px",  # ← gap between images (px)
                                                    "marginBottom": "4px"}))          # ← gap between rows (px)
    render_row = html.Div(rows_imgs, style={"marginBottom": "8px"})  # ← gap below image block

    # Metadata table
    rows = [("Stem",  stem), ("Site",  site)]
    if note:
        rows.append(("Notes", note))
    for col in META_COLS:
        if col in smeta:
            val = smeta[col]
            rows.append((col, val))

    def _row(k, v):
        is_url = str(v).startswith("http")
        cell = html.Td(
            html.A(v[:60] + ("…" if len(v) > 60 else ""), href=v, target="_blank",
                   style={"color": "#7cf"})
            if is_url else v,
            style={"color": "white", "paddingBottom": "3px",
                   "fontSize": "10px", "wordBreak": "break-word"}
        )
        return html.Tr([
            html.Td(k, style={"color": "#888", "fontSize": "10px",
                               "paddingRight": "8px", "whiteSpace": "nowrap",
                               "verticalAlign": "top", "paddingBottom": "3px"}),
            cell,
        ])

    meta_table = html.Table(
        html.Tbody([_row(k, v) for k, v in rows]),
        style={"width": "100%", "borderCollapse": "collapse"},
    )

    # Top-5 similar
    sims = sim[idx].copy()
    sims[idx] = -1
    top5 = np.argsort(sims)[::-1][:5]
    sim_items = [
        html.Li(
            f"{stems[j]}  ·  {site_map.get(stems[j], '?')}  ·  {sims[j]:.3f}",
            style={"fontSize": "10px", "color": "#ccc", "marginBottom": "2px"},
        )
        for j in top5
    ]

    return [
        html.H4(stem, style={"color": "#7cf", "margin": "0 0 4px 0",
                              "fontSize": "13px"}),
        html.P(site, style={"color": color.get(site, "#aaa"), "fontWeight": "bold",
                             "margin": "0 0 8px 0", "fontSize": "11px"}),
        render_row,
        html.Hr(style={"borderColor": "#2a2a3e", "margin": "6px 0"}),
        html.H5("Metadata", style={"color": "#aaa", "margin": "4px 0",
                                    "fontSize": "11px"}),
        meta_table,
        html.Hr(style={"borderColor": "#2a2a3e", "margin": "6px 0"}),
        html.H5("Most similar", style={"color": "#aaa", "margin": "4px 0",
                                        "fontSize": "11px"}),
        html.Ul(sim_items, style={"paddingLeft": "14px", "margin": "0"}),
    ]


def build_detail_archetype(k: int, stems: list[str], arch_S: np.ndarray,
                            site_map: dict, renders_dir: Path) -> list:
    """Detail panel for a clicked archetype vertex: show 3 closest stones."""
    arch_label = f"A{k + 1}"
    arch_names = {0: "Flat / Thin", 1: "Large · Round · Thick",
                  2: "Convex / Regular", 3: "Irregular · Pointed"}
    subtitle = arch_names.get(k, "")

    weights = arch_S[:, k]
    top3 = np.argsort(weights)[::-1][:3]

    # Three stone cards side by side; each card shows top + two side-plane renders
    ARCH_VIEWS = [("pZ", "Top"), ("pX", "Side"), ("pY", "Front")]
    cards = []
    for idx in top3:
        stem = stems[idx]
        site = site_map.get(stem, "Unknown")
        w    = weights[idx]

        thumb_row = []
        for rname, rlabel in ARCH_VIEWS:
            b64 = _b64(renders_dir / f"{stem}_{rname}.png")
            thumb_row.append(html.Div([
                html.P(rlabel, style={"fontSize": "8px", "color": "#666",
                                      "margin": "0 0 2px", "textAlign": "center"}),
                (html.Img(src=f"data:image/png;base64,{b64}",
                          style={"width": "100%", "borderRadius": "3px",
                                 "imageRendering": "pixelated", "display": "block"})
                 if b64 else html.Div("—", style={"color": "#333",
                                                   "textAlign": "center"})),
            ], style={"flex": "1 1 0"}))

        cards.append(html.Div([
            html.Div(thumb_row, style={"display": "flex", "gap": "3px",
                                       "marginBottom": "6px"}),
            html.P(stem,  style={"fontSize": "10px", "color": "#7cf",
                                  "margin": "0 0 2px", "textAlign": "center",
                                  "fontWeight": "bold"}),
            html.P(site,  style={"fontSize": "9px",  "color": "#aaa",
                                  "margin": "0",      "textAlign": "center"}),
            html.P(f"weight {w:.3f}", style={"fontSize": "9px", "color": "#00e5ff",
                                              "margin": "2px 0 0",
                                              "textAlign": "center"}),
        ], style={
            "flex": "1 1 0",
            "padding": "8px 6px",
            "backgroundColor": "#13132e",
            "borderRadius": "7px",
            "border": "1px solid #00e5ff33",
        }))

    return [
        html.H4(f"Archetype {arch_label}",
                style={"color": "#00e5ff", "margin": "0 0 2px", "fontSize": "14px"}),
        html.P(subtitle,
               style={"color": "#7799bb", "margin": "0 0 10px", "fontSize": "11px",
                      "fontStyle": "italic"}),
        html.P("3 stones closest to this archetype vertex",
               style={"color": "#555", "fontSize": "10px", "margin": "0 0 10px"}),
        html.Div(cards, style={"display": "flex", "gap": "8px", "marginBottom": "12px"}),
        html.Hr(style={"borderColor": "#2a2a3e", "margin": "6px 0"}),
        html.P("Click any stone in either PCA to restore the full detail view.",
               style={"color": "#444", "fontSize": "9px", "textAlign": "center"}),
    ]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def make_app(stems, E, coords, var, sim,
             site_map, notes_map, meta, renders_dir, run_dir,
             arch_Z, arch_S) -> dash.Dash:

    _, _, color = _site_colors(stems, site_map)

    pca3d_fig = build_pca3d(stems, coords, var, site_map, notes_map, arch_Z, arch_S)

    app = dash.Dash(__name__, title="Stone Tool Explorer")
    app.layout = html.Div(
        # ── page wrapper ─────────────────────────────────────────────────────
        # padding: outer whitespace around the whole page (top/bottom left/right)
        style={"backgroundColor": "#0f0f1a", "color": "white",
               "fontFamily": "ui-monospace, monospace",
               "minHeight": "100vh", "padding": "10px 14px"},
        children=[
            html.H2("Stone Tool Embedding Explorer",
                    style={"color": "#ccc", "marginBottom": "10px",
                           "fontSize": "16px"}),   # ← page title font size

            # ── top row: three panels side by side ───────────────────────────
            # gap: horizontal space between panels (px)
            # Each child's "flex" value controls its relative width:
            #   "1.5 1 0" means 1.5 parts  (three panels = 1.5 + 1.5 + 1.5 = 4.5 equal thirds)
            html.Div(style={"display": "flex", "gap": "10px",  # ← gap between panels (px)
                             "alignItems": "flex-start"}, children=[

                # ── PCA scatter (left) ────────────────────────────────────────
                # height set inside build_pca → _dark_layout(height=640)
                html.Div(style={"flex": "1.5 1 0", "minWidth": 0}, children=[
                    dcc.Graph(id="pca", figure=build_pca(stems, coords, var,
                                                         site_map, notes_map),
                              config={"displayModeBar": False}),
                ]),

                # ── Detail panel (centre) ────────────────────────────────────
                # padding: inner whitespace inside the panel box
                # minHeight: keeps the panel tall even when empty (before click)
                html.Div(id="detail",
                         style={"flex": "1.5 1 0", "minWidth": 0,
                                "backgroundColor": "#12122a",
                                "borderRadius": "8px", "padding": "10px",  # ← inner padding
                                "minHeight": "512px"}, children=[            # ← min panel height
                    html.P("Click a point in the PCA to inspect a stone.",
                           style={"color": "#555", "marginTop": "180px",
                                  "textAlign": "center"}),
                ]),

                # ── 3D PCA with archetype tetrahedron (right) ────────────────
                # Replaces the similarity matrix.
                # Click stone → detail panel updates; click A1-A4 → archetype panel.
                # height set inside build_pca3d (height=640)
                html.Div(style={"flex": "1.5 1 0", "minWidth": 0}, children=[
                    dcc.Graph(id="pca3d", figure=pca3d_fig,
                              config={"displayModeBar": False}),
                ]),
            ]),

            # sel stores the current selection:
            #   None                           → nothing selected
            #   {"type": "stone", "idx": N}    → stone N selected (from either PCA)
            #   {"type": "arch",  "k": K}      → archetype vertex K clicked (from 3D PCA)
            dcc.Store(id="sel", data=None),
        ],
    )

    # ── merge 2D and 3D click events into a single selection store ────────
    @app.callback(
        Output("sel", "data"),
        [Input("pca",   "clickData"),
         Input("pca3d", "clickData")],
        prevent_initial_call=True,
    )
    def _store(click_2d, click_3d):
        ctx = dash.callback_context
        if not ctx.triggered:
            return None
        triggered = ctx.triggered[0]["prop_id"]

        if "pca3d" in triggered and click_3d:
            pt   = click_3d["points"][0]
            text = pt.get("text", "")
            # Archetype vertex click: text is exactly "A1".."A4"
            if text in ("A1", "A2", "A3", "A4"):
                return {"type": "arch", "k": int(text[1]) - 1}
            # Stone click in 3D: customdata holds the original stone index
            cd = pt.get("customdata")
            if cd is not None:
                return {"type": "stone", "idx": int(cd)}

        elif "pca" in triggered and click_2d:
            return {"type": "stone", "idx": int(click_2d["points"][0]["customdata"])}

        return None

    # ── restore all sites when user clicks empty 2D plot area ────────────
    # Attaches a plotly_deselect listener each time the 2D PCA figure refreshes.
    # clicking empty plot space fires plotly_deselect (requires clickmode='event+select').
    app.clientside_callback(
        """
        function(figure) {
            setTimeout(function() {
                var gd = document.getElementById('pca');
                if (!gd) return;
                gd.removeAllListeners('plotly_deselect');
                gd.on('plotly_deselect', function() {
                    var n = gd.data.length;
                    Plotly.restyle(gd, {'visible': true},
                                   Array.from({length: n}, function(_, i) { return i; }));
                });
            }, 150);
            return window.dash_clientside.no_update;
        }
        """,
        Output('pca', 'className'),
        Input('pca', 'figure'),
    )

    # ── update detail panel and 2D PCA on any click ──────────────────────
    @app.callback(
        [Output("detail", "children"),
         Output("pca",    "figure")],
        Input("sel", "data"),
        prevent_initial_call=True,
    )
    def _update(sel):
        if sel is None:
            return (
                [html.P("Click a point.", style={"color": "#555"})],
                build_pca(stems, coords, var, site_map, notes_map),
            )

        if sel.get("type") == "arch":
            k = sel["k"]
            if arch_S is not None:
                detail = build_detail_archetype(k, stems, arch_S, site_map, renders_dir)
            else:
                detail = [html.P("Archetype data not found — run with --pca_only to generate.",
                                 style={"color": "#f88"})]
            # Don't change 2D PCA view for archetype clicks
            return detail, dash.no_update

        # Stone click (from 2D or 3D PCA)
        idx  = sel["idx"]
        stem = stems[idx]
        detail  = build_detail(stem, idx, stems, sim, site_map, notes_map,
                               meta, color, renders_dir)
        pca_fig = build_pca(stems, coords, var, site_map, notes_map,
                            selected_idx=idx)
        return detail, pca_fig

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--run_dir",     default=None,
                   help="Stamped run folder (default: most recent under --output_base)")
    p.add_argument("--output_base", default=str(OUTPUT_BASE))
    p.add_argument("--renders_dir", default=str(RENDERS_DIR))
    p.add_argument("--site_xlsx",   default=str(XLSX_SITE))
    p.add_argument("--meta_xlsx",   default=str(XLSX_META))
    p.add_argument("--port",        type=int, default=8050)
    p.add_argument("--debug",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = _latest_run(Path(args.output_base))
        if run_dir is None:
            sys.exit(f"No run folders found under {args.output_base}")
    log.info(f"Run: {run_dir}")

    stems, E, coords, var, sim, site_map, notes_map, meta, arch_Z, arch_S = load_all(
        run_dir, Path(args.renders_dir),
        Path(args.site_xlsx), Path(args.meta_xlsx),
    )
    log.info(f"{len(stems)} stones  D={E.shape[1]}")
    if arch_Z is not None:
        log.info(f"Archetype data loaded ({arch_Z.shape[0]} vertices)")
    else:
        log.info("No archetype data found — run dino_triptych.py --pca_only to generate")

    app = make_app(stems, E, coords, var, sim,
                   site_map, notes_map, meta, Path(args.renders_dir), run_dir,
                   arch_Z, arch_S)
    log.info(f"Dashboard → http://localhost:{args.port}")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
