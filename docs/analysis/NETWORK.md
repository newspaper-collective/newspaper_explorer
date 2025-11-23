# Entity Network Analysis

**"Six Degrees of Separation" for newspaper entities**

Analyze how entities are connected through their co-appearances on pages and dates, enabling discovery of entity relationships and connection paths.

## Overview

The Entity Network Analyzer builds a graph where:
- **Nodes** = entities (persons, locations, organizations, etc.)
- **Edges** = co-appearances (same page, same issue, or same date)
- **Edge weights** = number of co-occurrences

This enables you to:
1. Find paths between entities ("Six Degrees" style)
2. Discover which entities are most connected
3. Identify "bridge" entities that connect different groups
4. Export networks for visualization

## CLI Commands

### 1. Build Network and Show Statistics

```bash
# Build network from page co-occurrences
newspaper-explorer analyze entities network --source der_tag \
  --connection-type page --min-cooccur 2 --top-n 20

# Filter to specific entity types
newspaper-explorer analyze entities network --source der_tag \
  --entity-types "person,organization" --top-n 30

# Use different connection types
newspaper-explorer analyze entities network \
  --connection-type issue    # Same issue (more connections)
newspaper-explorer analyze entities network \
  --connection-type date     # Same date (most connections)
```

**Options:**
- `--connection-type`: How to define connections
  - `page`: Same page_id (strongest, fewest connections)
  - `issue`: Same issue_id (medium)
  - `date`: Same publication date (weakest, most connections)
- `--min-cooccur`: Minimum co-occurrences to create edge
- `--entity-types`: Filter to specific types (comma-separated)
- `--top-n`: Number of top entities to show

### 2. Find Path Between Entities

```bash
# Find shortest path between two entities
newspaper-explorer analyze entities find-path \
  --from "Kaiser Wilhelm II" --to "Berlin" --source der_tag

# Show detailed connection information
newspaper-explorer analyze entities find-path \
  --from "Bismarck" --to "Reichstag" --show-details

# Search with more degrees of separation
newspaper-explorer analyze entities find-path \
  --from "Berlin" --to "Paris" --max-degrees 10
```

**Options:**
- `--from`: Source entity name (case-sensitive)
- `--to`: Target entity name (case-sensitive)
- `--max-degrees`: Maximum degrees of separation to search (default: 6)
- `--show-details`: Show co-occurrence counts and dates
- `--connection-type`: Same as network command

### 3. Explore Entity Connections

```bash
# See all entities connected to a specific entity
newspaper-explorer analyze entities entity-connections \
  --entity "Kaiser Wilhelm II" --top-n 50

# Find frequently co-occurring entities
newspaper-explorer analyze entities entity-connections \
  --entity "Reichstag" --min-cooccur 5

# Explore locations
newspaper-explorer analyze entities entity-connections \
  --entity "Berlin" --min-cooccur 2 --top-n 30
```

**Options:**
- `--entity`: Entity name to explore (case-sensitive)
- `--min-cooccur`: Minimum co-occurrences to show
- `--top-n`: Number of connections to display

## Python API

### Basic Usage

```python
from newspaper_explorer.analyze.entities.network import EntityNetworkAnalyzer

# Create analyzer
analyzer = EntityNetworkAnalyzer(
    source_name="der_tag",
    method_id=None  # Uses most recent entity extraction
)

# Build network
stats = analyzer.build_network(
    connection_type="page",
    min_co_occurrences=2,
    entity_types=["person", "organization"]  # Optional filter
)

print(f"Network: {stats['nodes']} nodes, {stats['edges']} edges")
```

### Find Paths

```python
from newspaper_explorer.analyze.entities.network import find_entity_path

# Quick path finding
path = find_entity_path(
    source_entity="Kaiser Wilhelm II",
    target_entity="Berlin",
    connection_type="page"
)

if path:
    print(f"Found path: {path.degrees} degrees of separation")
    for entity in path.path:
        print(f"  → {entity}")
    
    # Access connection details
    for conn in path.connections:
        print(f"  Co-occurred {conn.co_occurrences} times")
        print(f"  On dates: {conn.dates}")
```

### Explore Connections

```python
# Get all connections for an entity
connections = analyzer.get_entity_connections(
    entity="Berlin",
    min_co_occurrences=3,
    sort_by="co_occurrences"
)

for conn in connections[:10]:
    other = conn.entity2 if conn.entity1 == "Berlin" else conn.entity1
    print(f"{other}: {conn.co_occurrences} co-occurrences")
```

### Network Statistics

```python
# Get top connected entities
top_entities = analyzer.get_top_entities(n=20, metric="degree")
for entity, degree in top_entities:
    print(f"{entity}: {int(degree)} connections")

# Get comprehensive stats
stats = analyzer.get_network_stats()
print(f"Average degree: {stats['avg_degree']:.2f}")
print(f"Max degree: {stats['max_degree']}")
print(f"Total co-occurrences: {stats['total_co_occurrences']}")
```

### Export for Visualization

```python
# Export to NetworkX (requires: pip install networkx)
G = analyzer.export_to_networkx()

# Calculate centrality measures
import networkx as nx
betweenness = nx.betweenness_centrality(G)
degree_centrality = nx.degree_centrality(G)

# Export to Gephi format (GEXF)
from pathlib import Path
analyzer.export_to_gephi(Path("output/entity_network.gexf"))
```

## Data Structures

### EntityConnection

Represents a connection between two entities:

```python
@dataclass
class EntityConnection:
    entity1: str              # First entity
    entity2: str              # Second entity
    connection_type: str      # "page", "issue", or "date"
    co_occurrences: int       # Number of times they co-occur
    dates: List[str]          # Dates they appeared together
    pages: List[str]          # Page IDs where they co-occur
    issues: List[str]         # Issue IDs where they co-occur
```

### ConnectionPath

Represents a path through the network:

```python
@dataclass
class ConnectionPath:
    source: str                         # Starting entity
    target: str                         # Target entity
    path: List[str]                     # Entities in path
    degrees: int                        # Length of path
    connections: List[EntityConnection] # Connections in path
```

## Use Cases

### 1. Historical Research

Find how historical figures are connected:

```bash
newspaper-explorer analyze entities find-path \
  --from "Bismarck" --to "Kaiser Wilhelm II"
```

### 2. Geographic Analysis

Explore which locations are mentioned together:

```bash
# Build network of locations
newspaper-explorer analyze entities network \
  --entity-types "location" --min-cooccur 3

# Find paths between cities
newspaper-explorer analyze entities find-path \
  --from "Berlin" --to "Paris"
```

### 3. Organization Networks

Discover organizational connections:

```bash
newspaper-explorer analyze entities network \
  --entity-types "organization" --top-n 30

newspaper-explorer analyze entities entity-connections \
  --entity "Reichstag" --min-cooccur 5
```

### 4. Event Analysis

Find entities connected to specific events:

```bash
newspaper-explorer analyze entities entity-connections \
  --entity "Krieg" --top-n 50
```

## Connection Types

The `--connection-type` parameter controls how connections are defined:

### Page (Strongest)
- **Definition**: Entities appear on the same page_id
- **Use**: Find entities that appear in close proximity
- **Result**: Fewer connections, strongest relationships
- **Example**: Two entities in the same article

### Issue (Medium)
- **Definition**: Entities appear in the same issue_id
- **Use**: Find entities mentioned in the same publication day
- **Result**: More connections than page, still contextually related
- **Example**: Two entities in different articles of same issue

### Date (Weakest)
- **Definition**: Entities appear on the same publication date
- **Use**: Find entities mentioned during the same time period
- **Result**: Most connections, weakest relationship
- **Example**: Two entities in different newspapers on same date

## Examples

See `src/newspaper_explorer/analyze/entities/examples_network.py` for complete examples:

```bash
cd /mnt/data/userfiles/westphal/newspaper_explorer
.venv/bin/python src/newspaper_explorer/analyze/entities/examples_network.py
```

## Requirements

**Core functionality** (included):
- polars
- Python 3.11+

**Optional (for advanced network analysis)**:
```bash
pip install networkx  # For NetworkX export and centrality measures
pip install matplotlib  # For visualizations
```

## Performance

Network building is fast:
- **67 entities**: ~instant
- **1000 entities**: ~1-2 seconds
- **10,000 entities**: ~10-20 seconds

Path finding uses BFS (breadth-first search):
- Very fast for most queries
- Handles disconnected networks gracefully

## Limitations

1. **Case-sensitive entity matching**: "Berlin" ≠ "berlin"
2. **Historical text variations**: "Deutſchland" ≠ "Deutschland"
3. **Entity extraction quality**: Depends on extraction method accuracy
4. **Memory usage**: Large networks (>50k entities) may need chunking

## Tips

1. **Start with pages**: Use `--connection-type page` for strongest relationships
2. **Filter by type**: Use `--entity-types` to focus on specific categories
3. **Adjust thresholds**: Lower `--min-cooccur` for broader networks
4. **Increase degrees**: Use `--max-degrees 10` if path not found
5. **Check entity names**: Use `network` command to see available entities
