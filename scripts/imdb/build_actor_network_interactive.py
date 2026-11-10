#!/usr/bin/env python3
"""Build an interactive HTML view of the principal-actor network.

This keeps the data-selection rules from build_actor_network.py, but writes a
self-contained HTML file. Hovering over an actor highlights that actor and all
of its visible collaborators; unrelated nodes and edges fade out.
"""

import argparse
import html
import json
import math
from pathlib import Path

from build_actor_network import (
    DEFAULT_DATA,
    DEFAULT_OUTPUT,
    build_network,
    load_cast,
    load_movies,
    load_names,
)


def calculate_centralities(node_ids, edge_weights):
    adjacency = {node_id: set() for node_id in node_ids}
    weighted = {node_id: {} for node_id in node_ids}
    for (actor_a, actor_b), weight in edge_weights.items():
        if actor_a not in adjacency or actor_b not in adjacency:
            continue
        adjacency[actor_a].add(actor_b)
        adjacency[actor_b].add(actor_a)
        weighted[actor_a][actor_b] = weight
        weighted[actor_b][actor_a] = weight

    degree = {node_id: len(adjacency[node_id]) for node_id in node_ids}
    closeness = {}
    betweenness = {node_id: 0.0 for node_id in node_ids}
    for source in node_ids:
        distances = {source: 0}
        predecessors = {node_id: [] for node_id in node_ids}
        queue = [source]
        for current in queue:
            for neighbour in adjacency[current]:
                if neighbour not in distances:
                    distances[neighbour] = distances[current] + 1
                    queue.append(neighbour)
                if distances[neighbour] == distances[current] + 1:
                    predecessors[neighbour].append(current)
        closeness[source] = (
            (len(distances) - 1) / sum(distances.values())
            if len(distances) > 1
            else 0.0
        )
        dependencies = {node_id: 0.0 for node_id in node_ids}
        for node_id in reversed(queue):
            for predecessor in predecessors[node_id]:
                dependencies[predecessor] += (
                    (1 + dependencies[node_id])
                    / len(predecessors[node_id])
                )
            if node_id != source:
                betweenness[node_id] += dependencies[node_id]
    scale = ((len(node_ids) - 1) * (len(node_ids) - 2)) or 1
    betweenness = {
        node_id: value / scale
        for node_id, value in betweenness.items()
    }

    eigenvector = {node_id: 1.0 for node_id in node_ids}
    for _ in range(100):
        next_values = {
            node_id: sum(
                weight * eigenvector[neighbour]
                for neighbour, weight in weighted[node_id].items()
            )
            for node_id in node_ids
        }
        norm = math.sqrt(sum(value * value for value in next_values.values())) or 1
        next_values = {
            node_id: value / norm for node_id, value in next_values.items()
        }
        if max(
            abs(next_values[node_id] - eigenvector[node_id])
            for node_id in node_ids
        ) < 1e-10:
            eigenvector = next_values
            break
        eigenvector = next_values

    clustering = {}
    for node_id in node_ids:
        neighbours = adjacency[node_id]
        possible = len(neighbours) * (len(neighbours) - 1) / 2
        links = sum(
            1
            for first in neighbours
            for second in neighbours
            if first < second and second in adjacency[first]
        )
        clustering[node_id] = links / possible if possible else 0.0
    return degree, closeness, betweenness, eigenvector, clustering


def write_html(output, names, movie_counts, edge_weights, plot_nodes):
    selected = {
        actor_id
        for actor_id, _ in sorted(
            movie_counts.items(), key=lambda item: (-item[1], names.get(item[0], item[0]))
        )[:plot_nodes]
    }
    edges = [
        {"a": a, "b": b, "weight": weight}
        for (a, b), weight in edge_weights.items()
        if a in selected and b in selected
    ]
    degree, closeness, betweenness, eigenvector, clustering = calculate_centralities(
        selected, edge_weights
    )
    weighted_degrees = {actor_id: 0 for actor_id in selected}
    collaborators = {actor_id: set() for actor_id in selected}
    for (actor_a, actor_b), weight in edge_weights.items():
        if actor_a in selected:
            weighted_degrees[actor_a] += weight
            collaborators[actor_a].add(actor_b)
        if actor_b in selected:
            weighted_degrees[actor_b] += weight
            collaborators[actor_b].add(actor_a)
    ranked = sorted(selected, key=lambda actor_id: (-movie_counts[actor_id], names.get(actor_id, actor_id)))
    width, height = 1400, 1000
    center_x, center_y = width / 2, height / 2
    radius = min(width, height) * 0.39
    positions = {}
    for index, actor_id in enumerate(ranked):
        angle = 2 * math.pi * index / max(1, len(ranked)) - math.pi / 2
        positions[actor_id] = {
            "x": round(center_x + radius * math.cos(angle), 1),
            "y": round(center_y + radius * math.sin(angle), 1),
        }

    nodes = [
        {
            "id": actor_id,
            "name": names.get(actor_id, actor_id),
            "movieCount": movie_counts[actor_id],
            "weightedDegree": weighted_degrees[actor_id],
            "collaboratorCount": len(collaborators[actor_id]),
            "degree": degree[actor_id],
            "closeness": round(closeness[actor_id], 4),
            "betweenness": round(betweenness[actor_id], 4),
            "eigenvector": round(eigenvector[actor_id], 4),
            "clustering": round(clustering[actor_id], 4),
            "x": positions[actor_id]["x"],
            "y": positions[actor_id]["y"],
        }
        for actor_id in ranked
    ]
    max_weight = max((edge["weight"] for edge in edges), default=1)
    max_movies = max((node["movieCount"] for node in nodes), default=1)
    for node in nodes:
        node["radius"] = round(4 + 12 * math.sqrt(node["movieCount"] / max_movies), 1)
    for edge in edges:
        edge["width"] = round(0.7 + 4.5 * math.sqrt(edge["weight"] / max_weight), 2)

    data = json.dumps({"nodes": nodes, "edges": edges}, separators=(",", ":"))
    title = html.escape("IMDb principal-actor co-appearance network")
    output.mkdir(parents=True, exist_ok=True)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ margin: 0; background: #f7f9fb; color: #17212b; font-family: system-ui, sans-serif; }}
  header {{ padding: 18px 28px 8px; }}
  h1 {{ margin: 0 0 6px; font-size: 24px; }}
  p {{ margin: 0 0 8px; color: #536471; }}
  .controls {{ display: flex; align-items: center; gap: 8px; margin-top: 12px; }}
  select {{ padding: 4px 8px; font: inherit; }}
  .content {{ display: grid; grid-template-columns: 230px minmax(0, 1400px) 420px; gap: 18px;
    max-width: 2100px; margin: auto; padding: 0 18px 18px; }}
  #network {{ display: block; width: 100%; height: auto; }}
  #centralitySummary {{ align-self: start; position: sticky; top: 18px; padding: 16px;
    background: white; border: 1px solid #d5dde5; border-radius: 8px; }}
  #centralitySummary h2 {{ margin: 0 0 8px; font-size: 17px; }}
  #centralitySummary p {{ margin: 0 0 12px; font-size: 12px; }}
  .centralityMetric {{ margin: 12px 0; font-size: 12px; }}
  .centralityMetric strong {{ display: block; margin-bottom: 3px; }}
  .centralityMetric a {{ color: #2563eb; }}
  #sidebar {{ align-self: start; position: sticky; top: 18px; }}
  #actorDetails {{ padding: 16px;
    background: white; border: 1px solid #d5dde5; border-radius: 8px; }}
  #actorDetails h2 {{ margin: 0 0 12px; font-size: 19px; }}
  #actorDetails p {{ margin: 0 0 12px; }}
  #actorDetails dl {{ margin: 0; }}
  #actorDetails dt {{ margin-top: 10px; color: #536471; font-size: 12px; }}
  #actorDetails dd {{ margin: 2px 0 0; }}
  #actorDetails a {{ color: #2563eb; }}
  #communityLegend {{ margin-top: 18px; padding-top: 14px; border-top: 1px solid #d5dde5; }}
  #communityLegend h3 {{ margin: 0; font-size: 15px; }}
  .legendHeader {{ display: flex; align-items: center; justify-content: space-between; gap: 6px; }}
  .legendHeader select {{ max-width: 190px; padding: 2px 4px; font-size: 12px; }}
  .legendItem {{ display: flex; align-items: center; gap: 7px; margin: 6px 0;
    padding: 3px; font-size: 12px; cursor: pointer; border-radius: 4px; }}
  .legendItem:hover {{ background: #eef4fa; }}
  .legendSwatch {{ width: 12px; height: 12px; border-radius: 50%; flex: 0 0 auto; }}
  .pinButton {{ margin-top: 14px; padding: 5px 8px; font: inherit; cursor: pointer; }}
  .edge {{ stroke: #9aa7b2; stroke-opacity: .55; }}
  .node {{ fill: #276fbf; opacity: .9; cursor: pointer; stroke: white; stroke-width: 1.2; }}
  .label {{ fill: #17212b; font-size: 12px; pointer-events: none; }}
  .faded {{ opacity: .07; }}
  .highlighted {{ opacity: 1; }}
  .tooltip {{ position: fixed; display: none; padding: 8px 10px; background: #17212b;
    color: white; border-radius: 5px; font-size: 13px; pointer-events: none; }}
  .legend {{ font-size: 13px; fill: #536471; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <p>Hover over an actor to highlight that actor and all visible collaborators. Node size = movies; edge width = shared movies.</p>
  <div class="controls">
    <label for="layout">Layout:</label>
    <select id="layout" aria-label="Choose network layout">
      <option value="force" selected>Force-directed</option>
      <option value="radial">Radial by movie count</option>
      <option value="grid">Grid</option>
      <option value="ring">Ring</option>
    </select>
    <label for="communities">Communities:</label>
    <select id="communities" aria-label="Choose community detection method">
      <option value="none">None</option>
      <option value="louvain" selected>Louvain</option>
      <option value="link">Link communities</option>
      <option value="components">Connected components</option>
    </select>
  </div>
  <p id="communityInfo" class="legend">Calculating Louvain communities...</p>
</header>
<div class="content">
<aside id="centralitySummary" aria-live="polite">
  <h2>Centrality extremes</h2>
  <p>Highest and lowest scores among the visible actors.</p>
  <div id="centralityMetrics"></div>
</aside>
<svg id="network" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"
     aria-label="{title}">
  <rect width="100%" height="100%" fill="white"/>
  <text x="30" y="40" font-size="24">IMDb actor co-appearance network</text>
  <text x="30" y="67" class="legend">Top {len(nodes)} actors by movie count; hover a node to isolate its visible connections</text>
  <g id="edges"></g>
  <g id="nodes"></g>
</svg>
<aside id="sidebar">
<div id="actorDetails" aria-live="polite">
  <h2>Actor details</h2>
  <p>Hover over an actor, or click one to keep its details here.</p>
</div>
  <div id="communityLegend" hidden></div>
</aside>
</div>
<div id="tooltip" class="tooltip"></div>
<script>
const graph = {data};
const svg = document.getElementById("network");
const edgeLayer = document.getElementById("edges");
const nodeLayer = document.getElementById("nodes");
const tooltip = document.getElementById("tooltip");
const actorDetails = document.getElementById("actorDetails");
const centralityMetrics = document.getElementById("centralityMetrics");
const communityLegend = document.getElementById("communityLegend");
const layoutSelect = document.getElementById("layout");
const communitySelect = document.getElementById("communities");
const communityInfo = document.getElementById("communityInfo");
const nodeById = new Map(graph.nodes.map(node => [node.id, node]));
const neighbours = new Map(graph.nodes.map(node => [node.id, new Set([node.id])]));
const nodeElements = new Map();
const labelElements = new Map();
const edgeElements = [];
const width = 1400;
const height = 1000;
const margin = 80;
const palette = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
  "#db2777", "#0891b2", "#65a30d", "#9333ea", "#ea580c", "#0f766e", "#be123c"];
let activeCommunities = new Map();
let activeColourByCommunity = new Map();
let pinnedActorId = null;
let pinnedCommunityId = null;
let communityEntries = [];
let communitySort = "movies";
const centralityFields = [
  ["degree", "Degree", value => value.toLocaleString()],
  ["closeness", "Closeness", value => value.toFixed(3)],
  ["betweenness", "Betweenness", value => value.toFixed(3)],
  ["eigenvector", "Eigenvector", value => value.toFixed(3)],
  ["clustering", "Clustering coefficient", value => value.toFixed(3)]
];

function renderCentralitySummary() {{
  centralityMetrics.innerHTML = centralityFields.map(([field, label, format]) => {{
    const sorted = [...graph.nodes].sort((first, second) => first[field] - second[field]);
    const lowest = sorted[0];
    const highest = sorted[sorted.length - 1];
    return `<div class="centralityMetric">
      <strong>${{label}}</strong>
      <span>Highest: <a href="#" data-actor="${{highest.id}}">${{highest.name}}</a>
        (${{format(highest[field])}})</span><br>
      <span>Lowest: <a href="#" data-actor="${{lowest.id}}">${{lowest.name}}</a>
        (${{format(lowest[field])}})</span>
    </div>`;
  }}).join("");
  centralityMetrics.querySelectorAll("a").forEach(link => link.addEventListener("click", event => {{
    event.preventDefault();
    const node = nodeById.get(event.currentTarget.dataset.actor);
    selectActor(node);
  }}));
}}

for (const edge of graph.edges) {{
  neighbours.get(edge.a).add(edge.b);
  neighbours.get(edge.b).add(edge.a);
  const a = nodeById.get(edge.a);
  const b = nodeById.get(edge.b);
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", a.x); line.setAttribute("y1", a.y);
  line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
  line.setAttribute("stroke-width", edge.width);
  line.dataset.a = edge.a; line.dataset.b = edge.b;
  line.classList.add("edge");
  edgeLayer.appendChild(line);
  edgeElements.push({{ edge, line }});
}}

function clamp(value, minimum, maximum) {{
  return Math.max(minimum, Math.min(maximum, value));
}}

function ringLayout() {{
  const radius = Math.min(width, height) * 0.39;
  const count = graph.nodes.length;
  return new Map(graph.nodes.map((node, index) => {{
    const angle = 2 * Math.PI * index / Math.max(1, count) - Math.PI / 2;
    return [node.id, {{
      x: width / 2 + radius * Math.cos(angle),
      y: height / 2 + radius * Math.sin(angle)
    }}];
  }}));
}}

function gridLayout() {{
  const columns = Math.ceil(Math.sqrt(graph.nodes.length * width / height));
  const rows = Math.ceil(graph.nodes.length / columns);
  const xStep = (width - 2 * margin) / Math.max(1, columns - 1);
  const yStep = (height - 2 * margin) / Math.max(1, rows - 1);
  return new Map(graph.nodes.map((node, index) => {{
    const column = index % columns;
    const row = Math.floor(index / columns);
    return [node.id, {{
      x: columns === 1 ? width / 2 : margin + column * xStep,
      y: rows === 1 ? height / 2 : margin + row * yStep
    }}];
  }}));
}}

function radialLayout() {{
  const positions = new Map();
  const rings = Math.max(1, Math.ceil((graph.nodes.length - 1) / 12));
  const outerRadius = Math.min(width, height) * 0.38;
  graph.nodes.forEach((node, index) => {{
    if (index === 0) {{
      positions.set(node.id, {{ x: width / 2, y: height / 2 }});
      return;
    }}
    const ring = Math.min(rings, Math.ceil(index / 12));
    const start = ring === 1 ? 1 : 1 + (ring - 1) * 12;
    const count = Math.min(12, graph.nodes.length - start);
    const angle = 2 * Math.PI * (index - start) / Math.max(1, count) - Math.PI / 2;
    const radius = rings === 1
      ? outerRadius
      : 130 + (ring - 1) * (outerRadius - 130) / (rings - 1);
    positions.set(node.id, {{
      x: width / 2 + radius * Math.cos(angle),
      y: height / 2 + radius * Math.sin(angle)
    }});
  }});
  return positions;
}}

function forceLayout() {{
  const positions = ringLayout();
  const iterations = 180;
  for (let iteration = 0; iteration < iterations; iteration++) {{
    const movement = new Map(graph.nodes.map(node => [node.id, {{ x: 0, y: 0 }}]));
    for (let i = 0; i < graph.nodes.length; i++) {{
      for (let j = i + 1; j < graph.nodes.length; j++) {{
        const a = graph.nodes[i];
        const b = graph.nodes[j];
        let dx = positions.get(a.id).x - positions.get(b.id).x;
        let dy = positions.get(a.id).y - positions.get(b.id).y;
        const distanceSquared = Math.max(25, dx * dx + dy * dy);
        const distance = Math.sqrt(distanceSquared);
        const force = 2400 / distanceSquared;
        dx = dx / distance * force;
        dy = dy / distance * force;
        movement.get(a.id).x += dx;
        movement.get(a.id).y += dy;
        movement.get(b.id).x -= dx;
        movement.get(b.id).y -= dy;
      }}
    }}
    for (const {{ edge }} of edgeElements) {{
      const a = positions.get(edge.a);
      const b = positions.get(edge.b);
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      const distance = Math.max(1, Math.sqrt(dx * dx + dy * dy));
      const target = 120 - 12 * Math.sqrt(edge.weight);
      const force = (distance - target) * 0.008;
      dx = dx / distance * force;
      dy = dy / distance * force;
      movement.get(edge.a).x += dx;
      movement.get(edge.a).y += dy;
      movement.get(edge.b).x -= dx;
      movement.get(edge.b).y -= dy;
    }}
    const cooling = 0.35 * (1 - iteration / iterations);
    for (const node of graph.nodes) {{
      const position = positions.get(node.id);
      const delta = movement.get(node.id);
      position.x = clamp(position.x + delta.x * cooling, margin, width - margin);
      position.y = clamp(position.y + delta.y * cooling, margin, height - margin);
    }}
  }}
  return positions;
}}

function updatePositions(positions) {{
  for (const node of graph.nodes) {{
    const position = positions.get(node.id);
    node.x = position.x;
    node.y = position.y;
    const circle = nodeElements.get(node.id);
    circle.setAttribute("cx", position.x);
    circle.setAttribute("cy", position.y);
    const label = labelElements.get(node.id);
    if (label) {{
      label.setAttribute("x", position.x + node.radius + 4);
      label.setAttribute("y", position.y + 4);
    }}
  }}
  for (const {{ edge, line }} of edgeElements) {{
    const a = positions.get(edge.a);
    const b = positions.get(edge.b);
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
  }}
}}

function applyLayout(layout) {{
  const positions = layout === "ring" ? ringLayout()
    : layout === "grid" ? gridLayout()
    : layout === "radial" ? radialLayout()
    : forceLayout();
  updatePositions(positions);
}}

function modularity(partition) {{
  const totalWeight = graph.edges.reduce((sum, edge) => sum + edge.weight, 0);
  if (!totalWeight) return 0;
  const degrees = new Map(graph.nodes.map(node => [node.id, 0]));
  for (const edge of graph.edges) {{
    degrees.set(edge.a, degrees.get(edge.a) + edge.weight);
    degrees.set(edge.b, degrees.get(edge.b) + edge.weight);
  }}
  const communityWeight = new Map();
  for (const node of graph.nodes) {{
    const community = partition.get(node.id);
    communityWeight.set(community, (communityWeight.get(community) || 0) + degrees.get(node.id));
  }}
  let score = 0;
  for (const edge of graph.edges) {{
    if (partition.get(edge.a) === partition.get(edge.b)) {{
      score += edge.weight / totalWeight;
    }}
  }}
  for (const weight of communityWeight.values()) {{
    score -= (weight * weight) / (4 * totalWeight * totalWeight);
  }}
  return score;
}}

function louvainCommunities() {{
  const partition = new Map(graph.nodes.map((node, index) => [node.id, index]));
  let bestScore = modularity(partition);
  let improved = true;
  let rounds = 0;
  while (improved && rounds++ < 25) {{
    improved = false;
    for (const node of graph.nodes) {{
      const current = partition.get(node.id);
      const candidates = new Set([current]);
      for (const edge of graph.edges) {{
        if (edge.a === node.id) candidates.add(partition.get(edge.b));
        if (edge.b === node.id) candidates.add(partition.get(edge.a));
      }}
      let bestCommunity = current;
      let nodeBestScore = bestScore;
      for (const candidate of candidates) {{
        if (candidate === current) continue;
        partition.set(node.id, candidate);
        const score = modularity(partition);
        if (score > nodeBestScore + 1e-10) {{
          nodeBestScore = score;
          bestCommunity = candidate;
        }}
      }}
      partition.set(node.id, bestCommunity);
      if (bestCommunity !== current) {{
        bestScore = nodeBestScore;
        improved = true;
      }}
    }}
  }}
  return {{ partition, score: modularity(partition) }};
}}

function connectedComponents() {{
  const partition = new Map();
  let component = 0;
  for (const start of graph.nodes) {{
    if (partition.has(start.id)) continue;
    const queue = [start.id];
    partition.set(start.id, component);
    while (queue.length) {{
      const nodeId = queue.pop();
      for (const neighbour of neighbours.get(nodeId)) {{
        if (!partition.has(neighbour)) {{
          partition.set(neighbour, component);
          queue.push(neighbour);
        }}
      }}
    }}
    component++;
  }}
  return {{ partition, score: modularity(partition) }};
}}

function linkCommunities() {{
  const incident = new Map(graph.nodes.map(node => [node.id, []]));
  graph.edges.forEach((edge, index) => {{
    incident.get(edge.a).push(index);
    incident.get(edge.b).push(index);
  }});
  const parent = graph.edges.map((edge, index) => index);
  function find(value) {{
    while (parent[value] !== value) {{
      parent[value] = parent[parent[value]];
      value = parent[value];
    }}
    return value;
  }}
  function join(a, b) {{
    const rootA = find(a);
    const rootB = find(b);
    if (rootA !== rootB) parent[rootB] = rootA;
  }}
  function similarity(first, second) {{
    const firstNeighbours = neighbours.get(first);
    const secondNeighbours = neighbours.get(second);
    let intersection = 0;
    for (const nodeId of firstNeighbours) {{
      if (secondNeighbours.has(nodeId)) intersection++;
    }}
    const union = firstNeighbours.size + secondNeighbours.size - intersection;
    return union ? intersection / union : 0;
  }}
  const threshold = 0.2;
  for (const edgeIds of incident.values()) {{
    for (let i = 0; i < edgeIds.length; i++) {{
      for (let j = i + 1; j < edgeIds.length; j++) {{
        const first = graph.edges[edgeIds[i]];
        const second = graph.edges[edgeIds[j]];
        const sharedNode = first.a === second.a || first.a === second.b
          ? first.a : first.b;
        const firstOther = first.a === sharedNode ? first.b : first.a;
        const secondOther = second.a === sharedNode ? second.b : second.a;
        if (similarity(firstOther, secondOther) >= threshold) {{
          join(edgeIds[i], edgeIds[j]);
        }}
      }}
    }}
  }}
  const groups = new Map();
  graph.edges.forEach((edge, index) => {{
    const root = find(index);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root).push(index);
  }});
  const nodeScores = new Map(graph.nodes.map(node => [node.id, new Map()]));
  for (const [root, edgeIds] of groups) {{
    for (const edgeId of edgeIds) {{
      const edge = graph.edges[edgeId];
      for (const nodeId of [edge.a, edge.b]) {{
        const scores = nodeScores.get(nodeId);
        scores.set(root, (scores.get(root) || 0) + edge.weight);
      }}
    }}
  }}
  const partition = new Map();
  for (const node of graph.nodes) {{
    const scores = nodeScores.get(node.id);
    partition.set(node.id, scores.size
      ? [...scores.entries()].sort((a, b) => b[1] - a[1])[0][0]
      : `isolated-${{node.id}}`);
  }}
  let triangles = 0;
  let possible = 0;
  for (const edgeIds of groups.values()) {{
    const nodeIds = new Set(edgeIds.flatMap(edgeId => {{
      const edge = graph.edges[edgeId];
      return [edge.a, edge.b];
    }}));
    const edgeCount = edgeIds.length;
    const maximum = nodeIds.size * (nodeIds.size - 1) / 2;
    if (maximum > nodeIds.size - 1) {{
      triangles += (edgeCount - (nodeIds.size - 1)) / (maximum - (nodeIds.size - 1));
      possible++;
    }}
  }}
  return {{
    partition,
    score: modularity(partition),
    linkCount: groups.size,
    density: possible ? triangles / possible : 0
  }};
}}

function colourCommunities(method) {{
  if (method === "none") {{
    activeCommunities = new Map();
    activeColourByCommunity = new Map();
    communityInfo.textContent = "Community colours are off.";
    communityLegend.hidden = true;
    pinnedCommunityId = null;
  }} else {{
    const result = method === "louvain" ? louvainCommunities()
      : method === "link" ? linkCommunities() : connectedComponents();
    activeCommunities = result.partition;
    const ids = [...new Set(activeCommunities.values())];
    activeColourByCommunity = new Map(ids.map((id, index) => [id, index]));
    communityLegend.hidden = false;
    communityEntries = ids.map(community => {{
      const members = graph.nodes
        .filter(node => activeCommunities.get(node.id) === community)
        .sort((first, second) => second.movieCount - first.movieCount);
      return {{
        id: community,
        members,
        leader: members[0],
        totalMovies: members.reduce((sum, node) => sum + node.movieCount, 0),
        centralityLeaders: Object.fromEntries(centralityFields.map(([field]) => {{
          return [field, members.reduce((best, node) =>
            node[field] > best[field] ? node : best, members[0])];
        }}))
      }};
    }});
    renderCommunityLegend();
    const metric = `modularity Q = ${{result.score.toFixed(3)}}`;
    communityInfo.textContent = method === "link"
      ? `Link communities: ${{result.linkCount}} link groups; ${{metric}} on the strongest-link node projection; partition density D = ${{result.density.toFixed(3)}}.`
      : `${{method === "louvain" ? "Louvain" : "Connected components"}}: ${{ids.length}} communities; ${{metric}}.`;
  }}
  for (const node of graph.nodes) {{
    const circle = nodeElements.get(node.id);
    const community = activeCommunities.get(node.id);
    circle.style.fill = community === undefined
      ? ""
      : palette[activeColourByCommunity.get(community) % palette.length];
  }}
}}

function renderCommunityLegend() {{
  communityLegend.innerHTML = `
    <div class="legendHeader">
      <h3>Communities</h3>
      <label>
        <span class="sr-only">Sort communities</span>
        <select id="communitySort" aria-label="Sort communities">
          <option value="movies">Most leader movies</option>
          <option value="actors">Most actors</option>
          <option value="degree">Highest degree centrality</option>
          <option value="closeness">Highest closeness centrality</option>
          <option value="betweenness">Highest betweenness centrality</option>
          <option value="eigenvector">Highest eigenvector centrality</option>
          <option value="clustering">Highest clustering coefficient</option>
        </select>
      </label>
    </div>`;
  const sortSelect = document.getElementById("communitySort");
  sortSelect.value = communitySort;
  sortSelect.addEventListener("change", event => {{
    communitySort = event.target.value;
    renderCommunityLegend();
  }});
  const sortedEntries = [...communityEntries].sort((first, second) =>
    communitySort === "actors"
      ? second.members.length - first.members.length
      : communitySort === "movies"
        ? second.leader.movieCount - first.leader.movieCount
        : second.centralityLeaders[communitySort][communitySort]
          - first.centralityLeaders[communitySort][communitySort]
  );
  for (const entry of sortedEntries) {{
      const item = document.createElement("div");
      item.className = "legendItem";
      item.dataset.community = entry.id;
      const swatch = document.createElement("span");
      swatch.className = "legendSwatch";
      swatch.style.backgroundColor = palette[activeColourByCommunity.get(entry.id) % palette.length];
      const label = document.createElement("span");
      const sortLeader = communitySort in entry.centralityLeaders
        ? entry.centralityLeaders[communitySort]
        : entry.leader;
      const sortValue = communitySort in entry.centralityLeaders
        ? `; ${{sortLeader.name}} ${{sortLeader[communitySort].toFixed(3)}}`
        : "";
      label.textContent = `${{entry.leader.name}} (${{entry.leader.movieCount}} movies; ${{entry.members.length}} actors${{sortValue}})`;
      item.append(swatch, label);
      item.addEventListener("mouseenter", () => {{
        highlightCommunity(entry.id);
        showCommunityDetails(entry);
      }});
      item.addEventListener("mouseleave", clearHighlight);
      item.addEventListener("click", () => selectCommunity(entry));
      communityLegend.appendChild(item);
  }}
}}

function showActorDetails(node, pinned = false) {{
  actorDetails.innerHTML = `
    <h2>${{node.name}}</h2>
    <dl>
      <dt>IMDb name ID</dt><dd><a href="https://www.imdb.com/name/${{node.id}}/" target="_blank" rel="noopener">${{node.id}}</a></dd>
      <dt>Movies in network</dt><dd>${{node.movieCount.toLocaleString()}}</dd>
      <dt>Collaborators</dt><dd>${{node.collaboratorCount.toLocaleString()}}</dd>
      <dt>Weighted degree</dt><dd>${{node.weightedDegree.toLocaleString()}} shared-movie links</dd>
      <dt>Degree centrality</dt><dd>${{node.degree.toLocaleString()}} direct collaborators</dd>
      <dt>Closeness centrality</dt><dd>${{node.closeness.toFixed(3)}}</dd>
      <dt>Betweenness centrality</dt><dd>${{node.betweenness.toFixed(3)}}</dd>
      <dt>Eigenvector centrality</dt><dd>${{node.eigenvector.toFixed(3)}}</dd>
      <dt>Clustering coefficient</dt><dd>${{node.clustering.toFixed(3)}}</dd>
    </dl>
    <button class="pinButton" type="button">${{pinned ? "Unpin details" : "Click actor to pin"}}</button>`;
  actorDetails.querySelector("button").addEventListener("click", () => {{
    pinnedActorId = pinnedActorId === node.id ? null : node.id;
    showActorDetails(node, pinnedActorId === node.id);
  }});
}}

function showCommunityDetails(entry, pinned = false) {{
  actorDetails.innerHTML = `
    <h2>${{entry.leader.name}}'s community</h2>
    <dl>
      <dt>Actors</dt><dd>${{entry.members.length.toLocaleString()}}</dd>
      <dt>Total movie appearances</dt><dd>${{entry.totalMovies.toLocaleString()}}</dd>
      <dt>Most prolific actor</dt><dd>${{entry.leader.name}} (${{entry.leader.movieCount}} movies)</dd>
    </dl>
    <button class="pinButton" type="button">${{pinned ? "Unpin community" : "Click to pin community"}}</button>`;
  actorDetails.querySelector("button").addEventListener("click", () => {{
    pinnedCommunityId = pinnedCommunityId === entry.id ? null : entry.id;
    showCommunityDetails(entry, pinnedCommunityId === entry.id);
  }});
}}

function showTooltip(event, node) {{
  tooltip.textContent = `${{node.name}} — ${{node.movieCount}} movies, ${{node.collaboratorCount}} collaborators`;
  tooltip.style.display = "block";
  tooltip.style.left = `${{event.clientX + 12}}px`;
  tooltip.style.top = `${{event.clientY + 12}}px`;
}}

function highlight(nodeId) {{
  const community = activeCommunities.get(nodeId);
  if (community === undefined) {{
    highlightNodes(neighbours.get(nodeId));
  }} else {{
    highlightCommunity(community);
  }}
}}

function highlightNodes(visible) {{
  for (const element of nodeLayer.children) {{
    element.classList.toggle("faded", !visible.has(element.dataset.id));
    element.classList.toggle("highlighted", visible.has(element.dataset.id));
  }}
  for (const element of edgeLayer.children) {{
    const connected = visible.has(element.dataset.a) && visible.has(element.dataset.b);
    element.classList.toggle("faded", !connected);
    element.classList.toggle("highlighted", connected);
  }}
}}

function highlightCommunity(community) {{
  const visible = new Set(graph.nodes
    .filter(node => activeCommunities.get(node.id) === community)
    .map(node => node.id));
  highlightNodes(visible);
}}

function clearHighlight() {{
  if (pinnedActorId !== null) {{
    highlight(pinnedActorId);
    return;
  }}
  if (pinnedCommunityId !== null) {{
    highlightCommunity(pinnedCommunityId);
    return;
  }}
  for (const element of [...nodeLayer.children, ...edgeLayer.children]) {{
    element.classList.remove("faded", "highlighted");
  }}
  tooltip.style.display = "none";
}}

function selectActor(node) {{
  pinnedCommunityId = null;
  pinnedActorId = node.id;
  highlight(node.id);
  showActorDetails(node, true);
}}

function selectCommunity(entry) {{
  pinnedActorId = null;
  pinnedCommunityId = entry.id;
  highlightCommunity(entry.id);
  showCommunityDetails(entry, true);
}}

for (const node of graph.nodes) {{
  const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
  circle.setAttribute("cx", node.x); circle.setAttribute("cy", node.y);
  circle.setAttribute("r", node.radius); circle.dataset.id = node.id;
  circle.classList.add("node");
  circle.addEventListener("mouseenter", event => {{
    highlight(node.id); showTooltip(event, node);
    if (pinnedActorId === null) showActorDetails(node);
  }});
  circle.addEventListener("mousemove", event => showTooltip(event, node));
  circle.addEventListener("mouseleave", clearHighlight);
  circle.addEventListener("click", () => selectActor(node));
  nodeLayer.appendChild(circle);
  nodeElements.set(node.id, circle);
  if (node.movieCount >= graph.nodes[0].movieCount * .35) {{
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", node.x + node.radius + 4);
    label.setAttribute("y", node.y + 4);
    label.textContent = node.name;
    label.classList.add("label");
    label.dataset.id = node.id;
    label.addEventListener("mouseenter", event => {{
      highlight(node.id); showTooltip(event, node);
      if (pinnedActorId === null) showActorDetails(node);
    }});
    label.addEventListener("mousemove", event => showTooltip(event, node));
    label.addEventListener("mouseleave", clearHighlight);
    label.addEventListener("click", () => selectActor(node));
    nodeLayer.appendChild(label);
    labelElements.set(node.id, label);
  }}
}}

layoutSelect.addEventListener("change", event => applyLayout(event.target.value));
communitySelect.addEventListener("change", event => colourCommunities(event.target.value));
renderCentralitySummary();
applyLayout(layoutSelect.value);
colourCommunities(communitySelect.value);
</script>
</body>
</html>
"""
    (output / "actor_network_interactive.html").write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-votes", type=int, default=10_000)
    parser.add_argument("--plot-nodes", type=int, default=150)
    args = parser.parse_args()

    movies = load_movies(args.data_dir, args.min_votes)
    cast = load_cast(args.data_dir, movies)
    names = load_names(args.data_dir, {actor for actors in cast.values() for actor in actors})
    movie_counts, edge_weights = build_network(movies, cast)
    write_html(args.output, names, movie_counts, edge_weights, args.plot_nodes)
    print(f"movies selected: {len(movies):,}")
    print(f"movies with at least two principal actors: {len(cast):,}")
    print(f"actors: {len(movie_counts):,}")
    print(f"visible actors: {min(args.plot_nodes, len(movie_counts)):,}")
    print(f"output: {args.output / 'actor_network_interactive.html'}")


if __name__ == "__main__":
    main()
