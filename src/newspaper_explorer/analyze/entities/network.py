"""
Entity network analysis: Six Degrees of Separation for newspaper entities.

Analyzes how entities are connected through co-appearances on pages and dates,
enabling discovery of entity relationships and connection paths.

Key features:
- Build entity co-occurrence networks from extraction results
- Find shortest paths between entities (degrees of separation)
- Calculate entity centrality and importance
- Export networks for visualization (NetworkX, Gephi, etc.)
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import polars as pl

from newspaper_explorer.config.base import get_config

logger = logging.getLogger(__name__)


@dataclass
class EntityConnection:
    """Connection between two entities."""

    entity1: str
    entity2: str
    connection_type: str  # 'page', 'issue', 'date'
    co_occurrences: int
    dates: List[str]
    pages: List[str]
    issues: List[str]


@dataclass
class ConnectionPath:
    """Path between two entities through the network."""

    source: str
    target: str
    path: List[str]
    degrees: int
    connections: List[EntityConnection]


class EntityNetworkAnalyzer:
    """
    Build and analyze entity co-occurrence networks.

    Creates a network graph where:
    - Nodes = entities
    - Edges = co-appearances (same page, same issue, or same date)
    - Edge weights = number of co-occurrences

    Example:
        ```python
        from newspaper_explorer.analyze.entities.network import EntityNetworkAnalyzer

        analyzer = EntityNetworkAnalyzer(
            source_name="der_tag",
            method_id="gliner_small_v2_1_20251109_230951"
        )

        # Build network from entity extractions
        analyzer.build_network(connection_type="page")

        # Find path between two entities
        path = analyzer.find_path("Kaiser Wilhelm II", "Berlin")
        print(f"Degrees of separation: {path.degrees}")

        # Get all connections for an entity
        connections = analyzer.get_entity_connections("Kaiser Wilhelm II")
        ```
    """

    def __init__(
        self,
        source_name: str = "der_tag",
        method_id: Optional[str] = None,
    ):
        """
        Initialize network analyzer.

        Args:
            source_name: Source dataset name (e.g., "der_tag").
            method_id: Entity extraction method ID. If None, uses most recent.
        """
        self.source_name = source_name
        self.method_id = method_id
        self.config = get_config()

        # Network data structures
        self.network: Dict[str, Dict[str, EntityConnection]] = defaultdict(dict)
        self.entity_metadata: Dict[str, Dict] = {}

        # Load data
        self._load_data()

    def _load_data(self):
        """Load entity extraction results and source data."""
        # Find method directory
        if self.method_id is None:
            entities_dir = self.config.results_dir / self.source_name / "entities"
            if not entities_dir.exists():
                raise ValueError(f"No entity results found for source '{self.source_name}'")

            # Get most recent method
            method_dirs = sorted([d for d in entities_dir.iterdir() if d.is_dir()])
            if not method_dirs:
                raise ValueError(f"No entity extraction methods found for '{self.source_name}'")
            self.method_id = method_dirs[-1].name
            logger.info(f"Using most recent method: {self.method_id}")

        # Load entity results
        entities_path = (
            self.config.results_dir
            / self.source_name
            / "entities"
            / self.method_id
            / "entities.parquet"
        )
        if not entities_path.exists():
            raise FileNotFoundError(f"Entity results not found: {entities_path}")

        logger.info(f"Loading entities from {entities_path}")
        self.entities_df = pl.read_parquet(entities_path)
        logger.info(f"Loaded {len(self.entities_df)} entity mentions")

        # Load source data for dates/pages
        source_path = (
            self.config.data_dir
            / "raw"
            / self.source_name
            / "text"
            / f"{self.source_name}_lines.parquet"
        )
        if not source_path.exists():
            raise FileNotFoundError(f"Source data not found: {source_path}")

        logger.info(f"Loading source data from {source_path}")
        self.source_df = pl.read_parquet(source_path).select(
            ["line_id", "date", "page_number", "page_id", "issue_id", "year", "month", "day"]
        )
        logger.info(f"Loaded {len(self.source_df)} source lines")

        # Join entities with source data to get dates/pages
        self.enriched_df = self.entities_df.join(
            self.source_df, left_on="line_id", right_on="line_id", how="left"
        )
        logger.info(f"Enriched entity data: {len(self.enriched_df)} rows")

    def build_network(
        self,
        connection_type: str = "page",
        min_co_occurrences: int = 1,
        entity_types: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        Build entity network from co-occurrences.

        Args:
            connection_type: How to define connections:
                - "page": Same page_id (strongest)
                - "issue": Same issue_id
                - "date": Same date (weakest, more connections)
            min_co_occurrences: Minimum co-occurrences for edge
            entity_types: Optional filter by entity types (e.g., ["person", "organization"])

        Returns:
            Statistics dict with network info
        """
        logger.info(f"Building network with connection_type='{connection_type}'")

        # Filter by entity types if specified
        df = self.enriched_df
        if entity_types:
            df = df.filter(pl.col("entity_type").is_in(entity_types))
            logger.info(f"Filtered to {len(df)} entities of types: {entity_types}")

        # Group by connection context (page/issue/date)
        if connection_type == "page":
            context_col = "page_id"
        elif connection_type == "issue":
            context_col = "issue_id"
        elif connection_type == "date":
            context_col = "date"
        else:
            raise ValueError(f"Invalid connection_type: {connection_type}")

        # Find co-occurring entities
        logger.info(f"Grouping by {context_col}...")

        # Get all entities per context
        context_entities = (
            df.group_by(context_col)
            .agg(
                [
                    pl.col("entity_text").unique().alias("entities"),
                    pl.col("date").first().alias("context_date"),
                    pl.col("page_id").first().alias("context_page_id"),
                    pl.col("issue_id").first().alias("context_issue_id"),
                ]
            )
            .filter(pl.col("entities").list.len() > 1)  # Only contexts with multiple entities
        )

        logger.info(f"Found {len(context_entities)} contexts with co-occurring entities")

        # Build edge list
        edges = defaultdict(
            lambda: {
                "co_occurrences": 0,
                "dates": set(),
                "pages": set(),
                "issues": set(),
            }
        )

        for row in context_entities.iter_rows(named=True):
            entities = row["entities"]
            date_str = str(row["context_date"].date()) if row["context_date"] else "unknown"
            page_id = row["context_page_id"] or "unknown"
            issue_id = row["context_issue_id"] or "unknown"

            # Create edges between all entity pairs in this context
            for i, entity1 in enumerate(entities):
                for entity2 in entities[i + 1 :]:
                    # Canonical edge key (alphabetically sorted)
                    edge_key = tuple(sorted([entity1, entity2]))

                    edges[edge_key]["co_occurrences"] += 1
                    edges[edge_key]["dates"].add(date_str)
                    edges[edge_key]["pages"].add(page_id)
                    edges[edge_key]["issues"].add(issue_id)

        # Filter by minimum co-occurrences and build network
        logger.info(f"Building network with min_co_occurrences={min_co_occurrences}")

        self.network.clear()
        filtered_edges = 0

        for (entity1, entity2), data in edges.items():
            if data["co_occurrences"] >= min_co_occurrences:
                connection = EntityConnection(
                    entity1=entity1,
                    entity2=entity2,
                    connection_type=connection_type,
                    co_occurrences=data["co_occurrences"],
                    dates=sorted(data["dates"]),
                    pages=sorted(data["pages"]),
                    issues=sorted(data["issues"]),
                )

                # Add bidirectional edges
                self.network[entity1][entity2] = connection
                self.network[entity2][entity1] = connection
                filtered_edges += 1

        # Calculate entity metadata
        self._calculate_entity_metadata()

        stats = {
            "nodes": len(self.network),
            "edges": filtered_edges,
            "connection_type": connection_type,
            "min_co_occurrences": min_co_occurrences,
            "avg_connections_per_entity": (
                sum(len(conns) for conns in self.network.values()) / len(self.network)
                if self.network
                else 0
            ),
        }

        logger.info(f"Network built: {stats['nodes']} nodes, {stats['edges']} edges")
        logger.info(f"Average connections per entity: {stats['avg_connections_per_entity']:.2f}")

        return stats

    def _calculate_entity_metadata(self):
        """Calculate metadata for each entity (degree, total co-occurrences, etc.)."""
        for entity, connections in self.network.items():
            self.entity_metadata[entity] = {
                "degree": len(connections),
                "total_co_occurrences": sum(c.co_occurrences for c in connections.values()),
                "connections": list(connections.keys()),
            }

    def find_path(
        self,
        source_entity: str,
        target_entity: str,
        max_degrees: int = 6,
    ) -> Optional[ConnectionPath]:
        """
        Find shortest path between two entities (BFS).

        Args:
            source_entity: Starting entity name
            target_entity: Target entity name
            max_degrees: Maximum degrees of separation to search

        Returns:
            ConnectionPath if found, None otherwise
        """
        if source_entity not in self.network:
            logger.warning(f"Source entity not found in network: '{source_entity}'")
            return None

        if target_entity not in self.network:
            logger.warning(f"Target entity not found in network: '{target_entity}'")
            return None

        if source_entity == target_entity:
            return ConnectionPath(
                source=source_entity,
                target=target_entity,
                path=[source_entity],
                degrees=0,
                connections=[],
            )

        # BFS to find shortest path
        from collections import deque

        queue = deque([(source_entity, [source_entity])])
        visited = {source_entity}

        while queue:
            current, path = queue.popleft()

            # Check depth limit
            if len(path) > max_degrees + 1:
                break

            # Check neighbors
            for neighbor in self.network[current].keys():
                if neighbor in visited:
                    continue

                new_path = path + [neighbor]

                if neighbor == target_entity:
                    # Found path! Build ConnectionPath
                    connections = []
                    for i in range(len(new_path) - 1):
                        entity1, entity2 = new_path[i], new_path[i + 1]
                        connections.append(self.network[entity1][entity2])

                    return ConnectionPath(
                        source=source_entity,
                        target=target_entity,
                        path=new_path,
                        degrees=len(new_path) - 1,
                        connections=connections,
                    )

                visited.add(neighbor)
                queue.append((neighbor, new_path))

        logger.info(f"No path found within {max_degrees} degrees")
        return None

    def get_entity_connections(
        self,
        entity: str,
        min_co_occurrences: int = 1,
        sort_by: str = "co_occurrences",
    ) -> List[EntityConnection]:
        """
        Get all connections for an entity.

        Args:
            entity: Entity name
            min_co_occurrences: Minimum co-occurrences to include
            sort_by: Sort by "co_occurrences" or "entity"

        Returns:
            List of EntityConnection objects
        """
        if entity not in self.network:
            logger.warning(f"Entity not found in network: '{entity}'")
            return []

        connections = [
            conn
            for conn in self.network[entity].values()
            if conn.co_occurrences >= min_co_occurrences
        ]

        if sort_by == "co_occurrences":
            connections.sort(key=lambda c: c.co_occurrences, reverse=True)
        elif sort_by == "entity":
            connections.sort(key=lambda c: c.entity2 if c.entity1 == entity else c.entity1)

        return connections

    def get_top_entities(
        self,
        n: int = 20,
        metric: str = "degree",
    ) -> List[Tuple[str, float]]:
        """
        Get top N entities by network metric.

        Args:
            n: Number of entities to return
            metric: Metric to rank by:
                - "degree": Number of connections
                - "total_co_occurrences": Sum of all co-occurrence counts

        Returns:
            List of (entity, score) tuples
        """
        if not self.entity_metadata:
            logger.warning("Network not built yet. Call build_network() first.")
            return []

        entities = [(entity, meta[metric]) for entity, meta in self.entity_metadata.items()]
        entities.sort(key=lambda x: x[1], reverse=True)

        return entities[:n]

    def export_to_networkx(self):
        """
        Export network to NetworkX graph.

        Returns:
            networkx.Graph object

        Requires:
            pip install networkx
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("NetworkX not installed. Install with: pip install networkx")

        G = nx.Graph()

        # Add nodes with metadata
        for entity, meta in self.entity_metadata.items():
            G.add_node(entity, **meta)

        # Add edges
        added_edges = set()
        for entity1, connections in self.network.items():
            for entity2, conn in connections.items():
                edge_key = tuple(sorted([entity1, entity2]))
                if edge_key not in added_edges:
                    G.add_edge(
                        entity1,
                        entity2,
                        weight=conn.co_occurrences,
                        connection_type=conn.connection_type,
                        dates=conn.dates,
                        pages=conn.pages,
                        issues=conn.issues,
                    )
                    added_edges.add(edge_key)

        logger.info(
            f"Exported NetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        return G

    def export_to_gephi(self, output_path: Path):
        """
        Export network to GEXF format for Gephi.

        Args:
            output_path: Path to save GEXF file

        Requires:
            pip install networkx
        """
        G = self.export_to_networkx()

        try:
            import networkx as nx

            nx.write_gexf(G, str(output_path))
            logger.info(f"Exported network to Gephi format: {output_path}")
        except Exception as e:
            logger.error(f"Failed to export to Gephi: {e}")
            raise

    def get_network_stats(self) -> Dict:
        """
        Get comprehensive network statistics.

        Returns:
            Dictionary with network metrics
        """
        if not self.network:
            return {"error": "Network not built"}

        degrees = [meta["degree"] for meta in self.entity_metadata.values()]

        return {
            "nodes": len(self.network),
            "edges": sum(len(conns) for conns in self.network.values()) // 2,
            "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
            "min_degree": min(degrees) if degrees else 0,
            "total_co_occurrences": sum(
                conn.co_occurrences
                for connections in self.network.values()
                for conn in connections.values()
            )
            // 2,  # Divide by 2 because edges are bidirectional
        }


def find_entity_path(
    source_entity: str,
    target_entity: str,
    source_name: str = "der_tag",
    method_id: Optional[str] = None,
    connection_type: str = "page",
    max_degrees: int = 6,
) -> Optional[ConnectionPath]:
    """
    Convenience function to find path between two entities.

    Args:
        source_entity: Starting entity name
        target_entity: Target entity name
        source_name: Source dataset name
        method_id: Entity extraction method ID (uses most recent if None)
        connection_type: "page", "issue", or "date"
        max_degrees: Maximum degrees of separation

    Returns:
        ConnectionPath if found, None otherwise

    Example:
        ```python
        from newspaper_explorer.analyze.entities.network import find_entity_path

        path = find_entity_path(
            source_entity="Kaiser Wilhelm II",
            target_entity="Berlin",
            connection_type="page"
        )

        if path:
            print(f"Found path with {path.degrees} degrees:")
            for entity in path.path:
                print(f"  → {entity}")
        ```
    """
    analyzer = EntityNetworkAnalyzer(source_name=source_name, method_id=method_id)
    analyzer.build_network(connection_type=connection_type)
    return analyzer.find_path(source_entity, target_entity, max_degrees=max_degrees)
