"""Build a self-contained HTML guide and static SVG previews from Mermaid sources."""
from __future__ import annotations

import html
import re
import textwrap
from pathlib import Path

import markdown
import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon


DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "architecture-guide.md"
DIAGRAMS = [
    ("dependencies", "模块依赖图"),
    ("first-training", "首次训练调用图"),
    ("inference", "聊天调用图"),
    ("resume-training", "断点恢复流程图"),
]


def node_from_reference(reference: str) -> tuple[str, str, str]:
    identifier_match = re.match(r"([A-Za-z][A-Za-z0-9]*)", reference.strip())
    if identifier_match is None:
        raise ValueError(f"Cannot parse Mermaid node: {reference}")
    identifier = identifier_match.group(1)
    quoted = re.search(r'"([^"]+)"', reference)
    label = quoted.group(1) if quoted else identifier
    if "{" in reference:
        shape = "decision"
    elif "[(" in reference or "([" in reference:
        shape = "artifact"
    else:
        shape = "process"
    return identifier, label, shape


def parse_mermaid(path: Path):
    nodes: dict[str, tuple[str, str]] = {}
    edges: list[tuple[str, str, str]] = []
    styles: dict[str, tuple[str, str]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("flowchart ", "subgraph ")) or line == "end":
            continue
        if line.startswith("style "):
            match = re.match(
                r"style\s+(\w+)\s+fill:(#[0-9A-Fa-f]{6}),stroke:(#[0-9A-Fa-f]{6})",
                line,
            )
            if match:
                styles[match.group(1)] = (match.group(2), match.group(3))
            continue
        match = re.match(
            r"(.+?)\s+(-->|-.->|==>)\s*(?:\|\"?(.+?)\"?\|\s*)?(.+)$",
            line,
        )
        if match is None:
            continue
        source_id, source_label, source_shape = node_from_reference(match.group(1))
        target_id, target_label, target_shape = node_from_reference(match.group(4))
        nodes.setdefault(source_id, (source_label, source_shape))
        nodes.setdefault(target_id, (target_label, target_shape))
        edge_label = (match.group(3) or "").strip('"')
        edges.append((source_id, target_id, edge_label))
    return nodes, edges, styles


def layered_positions(nodes, edges):
    layout_graph = nx.DiGraph()
    layout_graph.add_nodes_from(nodes)
    cycle_edges = set()
    for source, target, _ in edges:
        layout_graph.add_edge(source, target)
        if not nx.is_directed_acyclic_graph(layout_graph):
            layout_graph.remove_edge(source, target)
            cycle_edges.add((source, target))
    generations = list(nx.topological_generations(layout_graph))
    positions = {}
    for layer, generation in enumerate(generations):
        ordered = list(generation)
        count = len(ordered)
        for index, identifier in enumerate(ordered):
            x = index - (count - 1) / 2
            positions[identifier] = (x * 3.0, -layer * 2.0)
    return positions, cycle_edges


def render_diagram(source: Path, destination: Path, title: str) -> str:
    nodes, edges, styles = parse_mermaid(source)
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((source_id, target_id) for source_id, target_id, _ in edges)
    positions, cycle_edges = layered_positions(nodes, edges)
    widest = max((sum(1 for _, y in positions.values() if y == level) for level in {p[1] for p in positions.values()}), default=1)
    height = max(7.0, len({position[1] for position in positions.values()}) * 1.45)
    figure, axis = plt.subplots(figsize=(max(12.0, widest * 3.4), height))
    axis.set_title(
        title,
        fontsize=18,
        pad=18,
        fontweight="bold",
        fontfamily="Arial Unicode MS",
    )
    axis.axis("off")

    # Draw connectors first so that node cards cover the line ends cleanly.
    for source_id, target_id, edge_label in edges:
        source_position = positions[source_id]
        target_position = positions[target_id]
        is_cycle = (source_id, target_id) in cycle_edges
        connector = FancyArrowPatch(
            source_position,
            target_position,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            linestyle="--" if is_cycle else "-",
            color="#A66A35" if is_cycle else "#607D8B",
            connectionstyle="arc3,rad=0.22" if is_cycle else "arc3,rad=0",
            shrinkA=30,
            shrinkB=30,
            zorder=1,
        )
        axis.add_patch(connector)
        if edge_label:
            middle_x = (source_position[0] + target_position[0]) / 2
            middle_y = (source_position[1] + target_position[1]) / 2 + 0.13
            axis.text(
                middle_x,
                middle_y,
                edge_label,
                ha="center",
                va="center",
                fontsize=8,
                fontfamily="Arial Unicode MS",
                bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none"},
                zorder=4,
            )

    for identifier, (label, shape) in nodes.items():
        x, y = positions[identifier]
        wrap_target = label.replace(".", ". ").replace("：", "： ")
        wrapped = "\n".join(
            line.strip()
            for line in textwrap.wrap(
                wrap_target,
                width=18,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        line_count = wrapped.count("\n") + 1
        width = 2.6
        height = 0.78 + (line_count - 1) * 0.22
        fill, border = styles.get(identifier, ("#EAF2F8", "#5B7890"))
        if shape == "decision":
            patch = Polygon(
                [(x, y + height / 2), (x + width / 2, y),
                 (x, y - height / 2), (x - width / 2, y)],
                closed=True,
                facecolor=fill,
                edgecolor=border,
                linewidth=1.7,
                zorder=2,
            )
        elif shape == "artifact":
            patch = Ellipse(
                (x, y),
                width=width,
                height=height,
                facecolor=fill,
                edgecolor=border,
                linewidth=1.7,
                zorder=2,
            )
        else:
            patch = FancyBboxPatch(
                (x - width / 2, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.04,rounding_size=0.07",
                facecolor=fill,
                edgecolor=border,
                linewidth=1.7,
                zorder=2,
            )
        axis.add_patch(patch)
        axis.text(
            x,
            y,
            wrapped,
            ha="center",
            va="center",
            fontsize=8.4,
            fontfamily="Arial Unicode MS",
            zorder=3,
        )

    x_values = [position[0] for position in positions.values()]
    y_values = [position[1] for position in positions.values()]
    axis.set_xlim(min(x_values) - 1.7, max(x_values) + 1.7)
    axis.set_ylim(min(y_values) - 1.0, max(y_values) + 1.0)
    axis.set_aspect("equal", adjustable="box")
    figure.tight_layout()
    figure.savefig(destination, format="svg", bbox_inches="tight", facecolor="white")
    figure.savefig(
        destination.with_suffix(".png"),
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    svg = destination.read_text(encoding="utf-8")
    return svg[svg.index("<svg") :]


def main() -> None:
    matplotlib.rcParams["svg.fonttype"] = "none"
    rendered = {}
    for stem, title in DIAGRAMS:
        rendered[stem] = render_diagram(
            DOCS / "diagrams" / f"{stem}.mmd",
            DOCS / "diagrams" / f"{stem}.svg",
            title,
        )

    markdown_source = SOURCE.read_text(encoding="utf-8")
    index = 0

    def replace_mermaid(match):
        nonlocal index
        stem, title = DIAGRAMS[index]
        index += 1
        source = match.group(1).strip()
        return (
            f'<figure class="diagram" aria-label="{html.escape(title)}">'
            f'{rendered[stem]}'
            f'<figcaption>{html.escape(title)}。'
            '<details><summary>查看 Mermaid 源码</summary>'
            f'<pre><code>{html.escape(source)}</code></pre>'
            '</details></figcaption></figure>'
        )

    markdown_source = re.sub(
        r"```mermaid\s*\n(.*?)```",
        replace_mermaid,
        markdown_source,
        flags=re.DOTALL,
    )
    if index != len(DIAGRAMS):
        raise ValueError(f"Expected {len(DIAGRAMS)} Mermaid diagrams, found {index}")
    body = markdown.markdown(
        "[TOC]\n\n" + markdown_source,
        extensions=["extra", "toc", "sane_lists"],
        output_format="html5",
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI-MODEL 架构与源码阅读手册</title>
<style>
:root {{ color-scheme: light; --ink:#18212b; --muted:#5d6875; --line:#d8e0e8; --accent:#1769aa; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#f5f7fa; color:var(--ink); font:16px/1.72 -apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans CJK SC",sans-serif; }}
main {{ max-width:1120px; margin:0 auto; background:white; padding:48px clamp(24px,5vw,72px) 80px; box-shadow:0 0 32px rgba(32,48,64,.08); }}
h1 {{ font-size:2.15rem; line-height:1.25; }} h2 {{ margin-top:2.5em; border-bottom:1px solid var(--line); padding-bottom:.35em; }}
h3 {{ margin-top:2em; }} a {{ color:var(--accent); }} code {{ font-family:"SFMono-Regular",Consolas,monospace; }}
pre {{ overflow:auto; background:#f3f6f8; border:1px solid var(--line); border-radius:8px; padding:16px; line-height:1.5; }}
table {{ border-collapse:collapse; width:100%; margin:1.2em 0; }} th,td {{ border:1px solid var(--line); padding:9px 12px; vertical-align:top; }} th {{ background:#edf3f8; text-align:left; }}
.toc {{ background:#f3f7fb; border:1px solid #cdddea; border-radius:10px; padding:12px 24px; margin:24px 0 36px; }}
.toc > ul {{ columns:2; }} .diagram {{ margin:28px 0; padding:16px; border:1px solid var(--line); border-radius:12px; background:#fff; }}
.diagram svg {{ display:block; width:100%; height:auto; }} figcaption {{ color:var(--muted); margin-top:10px; }} details {{ margin-top:8px; }}
blockquote {{ border-left:4px solid #86afd0; margin-left:0; padding-left:16px; color:var(--muted); }}
@media (max-width:700px) {{ main {{ padding:28px 18px 56px; }} .toc > ul {{ columns:1; }} table {{ font-size:.88rem; }} }}
@media print {{ body {{ background:white; }} main {{ box-shadow:none; max-width:none; }} .toc {{ break-after:page; }} .diagram {{ break-inside:avoid; }} }}
</style>
</head>
<body><main>{body}</main></body>
</html>
"""
    (DOCS / "architecture-guide.html").write_text(document, encoding="utf-8")
    print(f"Wrote {DOCS / 'architecture-guide.html'} with {index} static diagrams")


if __name__ == "__main__":
    main()
