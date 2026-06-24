"""
Interactive Stone Tool Embedding Dashboard.

Layout
------
  Top-left   : PCA scatter  (click a point to inspect)
  Top-centre : Stone detail panel  (renders + metadata + similar stones)
  Top-right  : Cosine similarity matrix  (selected row/col highlighted)
  Bottom     : Site map placeholder  (populate SITE_COORDS to activate)

Usage
-----
  python scripts/dashboard.py
  python scripts/dashboard.py --run_dir outputs/dino_7ch_v2/20260624_123456
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

OUTPUT_BASE = ROOT / "outputs" / "dino_7ch_v2"
RENDERS_DIR = ROOT / "outputs" / "renders"
XLSX_SITE   = ROOT / "wrl" / "Handaxes 2026 list with sites.xlsx"
XLSX_META   = ROOT / "wrl" / "Handaxes2026list_with_sites_and_metadata.xlsx"

RENDER_SLOTS = [
    ("top",      "Top normals"),
    ("bot",      "Bottom normals"),
    ("thick",    "Thickness"),
    ("dihedral", "Dihedral"),
]

# ---------------------------------------------------------------------------
# Site coordinates  (lat, lon)
# Add entries here to activate the map — keyed by the site-name prefix used
# in site_map (text before the first '(').
# ---------------------------------------------------------------------------
SITE_COORDS: dict[str, tuple[float, float]] = {
    # "Emek Refaim":          (31.760, 35.205),
    # "Gesher Benot Ya'akov": (33.009, 35.629),
    # "Holon":                (32.011, 34.778),
    # "Maayan Baruch":        (33.215, 35.623),
    # "Zihor":                (30.585, 35.046),
    # "Jaljulia":             (32.158, 34.945),
    # "Ubeidiya":             (32.687, 35.553),
}

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
        # First column = site name; remaining = metadata
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

    return stems, E, coords, var, sim, site_map, notes_map, meta


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

def _dark_layout(**kw) -> dict:
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
            marker=dict(size=9, color=color[site],
                        line=dict(width=1, color="white")),
            text=[stems[i] for i in idxs],
            customdata=idxs,
            hovertemplate="<b>%{text}</b><br>" + site + "<extra></extra>",
        ))

    if selected_idx is not None:
        fig.add_trace(go.Scatter(
            x=[coords[selected_idx, 0]], y=[coords[selected_idx, 1]],
            mode="markers", name="selected", showlegend=False,
            marker=dict(size=18, color="rgba(0,0,0,0)",
                        line=dict(width=2.5, color="yellow")),
            hoverinfo="skip",
        ))

    n_pc = len(var)
    fig.update_layout(
        **_dark_layout(height=460),
        title=f"PCA — PC1 {var[0]:.1%} | PC2 {var[1]:.1%}",
        xaxis=dict(title=f"PC1 ({var[0]:.1%})", gridcolor="#222", color="#aaa"),
        yaxis=dict(title=f"PC2 ({var[1]:.1%})", gridcolor="#222", color="#aaa"),
        legend=dict(bgcolor="#1a1a2e", bordercolor="#333", font_size=9),
        clickmode="event",
    )
    return fig


def build_sim_matrix(stems, sim, selected_idx: int | None = None) -> go.Figure:
    labels = [s[:14] for s in stems]
    shapes = []
    if selected_idx is not None:
        n = len(stems)
        for rect in [
            dict(x0=-0.5, x1=n - 0.5, y0=selected_idx - 0.5, y1=selected_idx + 0.5),
            dict(x0=selected_idx - 0.5, x1=selected_idx + 0.5, y0=-0.5, y1=n - 0.5),
        ]:
            shapes.append(dict(type="rect", xref="x", yref="y",
                               fillcolor="rgba(255,255,0,0.12)",
                               line=dict(color="yellow", width=1), **rect))

    fig = go.Figure(go.Heatmap(
        z=sim, x=labels, y=labels,
        colorscale="Plasma", zmin=0, zmax=1,
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>sim = %{z:.3f}<extra></extra>",
    ))
    layout = _dark_layout(height=460)
    layout["margin"] = dict(l=90, r=10, t=45, b=90)
    fig.update_layout(
        **layout,
        title="Cosine similarity",
        xaxis=dict(tickangle=45, tickfont_size=7, color="#aaa"),
        yaxis=dict(tickfont_size=7, color="#aaa", autorange="reversed"),
        shapes=shapes,
    )
    return fig


def build_map(unique_sites: list[str], color: dict,
              highlight: str | None = None) -> go.Figure:
    fig = go.Figure()

    known = {s: SITE_COORDS[_site_key(s)] for s in unique_sites
             if _site_key(s) in SITE_COORDS}
    if known:
        for site, (lat, lon) in known.items():
            key  = _site_key(site)
            size = 18 if key == _site_key(highlight or "") else 10
            bord = "yellow" if key == _site_key(highlight or "") else "white"
            fig.add_trace(go.Scattergeo(
                lat=[lat], lon=[lon], text=[site], mode="markers+text",
                textposition="top center",
                marker=dict(size=size, color=color.get(site, "#aaa"),
                            line=dict(width=2, color=bord)),
                showlegend=False,
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))
    else:
        fig.add_annotation(
            text="⚑  Add lat/lon to SITE_COORDS in dashboard.py to activate this map",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=12, color="#666"),
        )

    hl_text = f"  —  highlighted: {highlight}" if highlight else ""
    fig.update_layout(
        **_dark_layout(height=220, margin=dict(l=0, r=0, t=35, b=0)),
        title=f"Site map{hl_text}",
        geo=dict(
            bgcolor="#0f0f1a", landcolor="#1a2a1a", lakecolor="#111122",
            showland=True, showlakes=True, showcoastlines=True,
            coastlinecolor="#333", showframe=False,
            projection_type="natural earth",
            center=dict(lat=32, lon=35) if not known else {},
            projection_scale=8 if not known else 1,
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------

def _render_img(b64: str | None, label: str) -> html.Div:
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
        html.P(label, style={"fontSize": "9px", "color": "#888",
                              "margin": "0 0 2px 0", "textAlign": "center"}),
        child,
    ], style={"flex": "1 1 0"})


def build_detail(stem: str, idx: int, stems: list[str], sim: np.ndarray,
                 site_map: dict, notes_map: dict, meta: dict,
                 color: dict, renders_dir: Path) -> list:
    site  = site_map.get(stem, "Unknown")
    note  = notes_map.get(stem, "")
    skey  = _site_key(site)
    smeta = meta.get(skey, {})

    # Render images
    imgs = [_render_img(_b64(renders_dir / f"{stem}_{rname}.png"), label)
            for rname, label in RENDER_SLOTS]
    render_row = html.Div(imgs, style={"display": "flex", "gap": "4px",
                                        "marginBottom": "10px"})

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


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def make_app(stems, E, coords, var, sim,
             site_map, notes_map, meta, renders_dir) -> dash.Dash:

    stem_sites, unique_sites, color = _site_colors(stems, site_map)

    app = dash.Dash(__name__, title="Stone Tool Explorer")
    app.layout = html.Div(
        style={"backgroundColor": "#0f0f1a", "color": "white",
               "fontFamily": "ui-monospace, monospace",
               "minHeight": "100vh", "padding": "10px 14px"},
        children=[
            html.H2("Stone Tool Embedding Explorer",
                    style={"color": "#ccc", "marginBottom": "10px",
                           "fontSize": "16px"}),

            # ── top row ──────────────────────────────────────────────────
            html.Div(style={"display": "flex", "gap": "10px",
                             "alignItems": "flex-start"}, children=[

                # PCA
                html.Div(style={"flex": "2 1 0", "minWidth": 0}, children=[
                    dcc.Graph(id="pca", figure=build_pca(stems, coords, var,
                                                         site_map, notes_map),
                              config={"displayModeBar": False}),
                ]),

                # Detail panel
                html.Div(id="detail",
                         style={"flex": "1.4 1 0", "minWidth": 0,
                                "backgroundColor": "#12122a",
                                "borderRadius": "8px", "padding": "10px",
                                "minHeight": "460px"}, children=[
                    html.P("Click a point in the PCA to inspect a stone.",
                           style={"color": "#555", "marginTop": "180px",
                                  "textAlign": "center"}),
                ]),

                # Similarity matrix
                html.Div(style={"flex": "2 1 0", "minWidth": 0}, children=[
                    dcc.Graph(id="sim", figure=build_sim_matrix(stems, sim),
                              config={"displayModeBar": False}),
                ]),
            ]),

            # ── map ──────────────────────────────────────────────────────
            html.Div(style={"marginTop": "10px"}, children=[
                dcc.Graph(id="map",
                          figure=build_map(unique_sites, color),
                          config={"displayModeBar": False}),
            ]),

            dcc.Store(id="sel", data=None),
        ],
    )

    # ── store click index ─────────────────────────────────────────────────
    @app.callback(Output("sel", "data"), Input("pca", "clickData"),
                  prevent_initial_call=True)
    def _store(click_data):
        if not click_data:
            return None
        return int(click_data["points"][0]["customdata"])

    # ── update everything from stored index ───────────────────────────────
    @app.callback(
        [Output("detail", "children"),
         Output("sim",    "figure"),
         Output("map",    "figure"),
         Output("pca",    "figure")],
        Input("sel", "data"),
        prevent_initial_call=True,
    )
    def _update(idx):
        if idx is None:
            return (
                [html.P("Click a point.", style={"color": "#555"})],
                build_sim_matrix(stems, sim),
                build_map(unique_sites, color),
                build_pca(stems, coords, var, site_map, notes_map),
            )

        stem = stems[idx]
        site = site_map.get(stem, "Unknown")

        detail  = build_detail(stem, idx, stems, sim, site_map, notes_map,
                               meta, color, renders_dir)
        sim_fig = build_sim_matrix(stems, sim, selected_idx=idx)
        map_fig = build_map(unique_sites, color, highlight=site)
        pca_fig = build_pca(stems, coords, var, site_map, notes_map,
                            selected_idx=idx)

        return detail, sim_fig, map_fig, pca_fig

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

    stems, E, coords, var, sim, site_map, notes_map, meta = load_all(
        run_dir, Path(args.renders_dir),
        Path(args.site_xlsx), Path(args.meta_xlsx),
    )
    log.info(f"{len(stems)} stones  D={E.shape[1]}")

    app = make_app(stems, E, coords, var, sim,
                   site_map, notes_map, meta, Path(args.renders_dir))
    log.info(f"Dashboard → http://localhost:{args.port}")
    app.run(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
