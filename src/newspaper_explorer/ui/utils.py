"""
Utility functions for loading and processing historical newspaper data
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from neo4j import GraphDatabase
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ============================================================================
# Neo4j Functions
# ============================================================================

class Neo4jConnector:
    """Connect to Neo4j and query newspaper data"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def get_entity_network(self, start_date: str, end_date: str, limit: int = 100):
        """Get entity relationships within date range"""
        query = """
        MATCH (d:Date)-[:CONTAINS]->(tb:TextBlock)-[:MENTIONS]->(e:Entity)
        WHERE d.date >= $start_date AND d.date <= $end_date
        RETURN d.date as date, 
               tb.id as textblock_id, 
               e.name as entity, 
               e.type as entity_type,
               tb.text as text
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, 
                               start_date=start_date, 
                               end_date=end_date, 
                               limit=limit)
            return [dict(record) for record in result]
    
    def get_entity_cooccurrence(self, entity_type: str = None):
        """Get entities that co-occur in the same textblocks"""
        query = """
        MATCH (e1:Entity)<-[:MENTIONS]-(tb:TextBlock)-[:MENTIONS]->(e2:Entity)
        WHERE id(e1) < id(e2)
        """ + (f"AND e1.type = '{entity_type}' AND e2.type = '{entity_type}'" if entity_type else "") + """
        RETURN e1.name as entity1, 
               e2.name as entity2, 
               count(tb) as cooccurrence_count
        ORDER BY cooccurrence_count DESC
        LIMIT 100
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]
    
    def get_entity_timeline(self, entity_name: str):
        """Get timeline of mentions for a specific entity"""
        query = """
        MATCH (d:Date)-[:CONTAINS]->(tb:TextBlock)-[:MENTIONS]->(e:Entity)
        WHERE e.name = $entity_name
        RETURN d.date as date, count(tb) as mention_count
        ORDER BY d.date
        """
        
        with self.driver.session() as session:
            result = session.run(query, entity_name=entity_name)
            return pd.DataFrame([dict(record) for record in result])
    
    def get_concept_network(self, concept_type: str = None):
        """Get concept relationships"""
        query = """
        MATCH (d:Date)-[:CONTAINS]->(c:Concept)
        """ + (f"WHERE c.type = '{concept_type}'" if concept_type else "") + """
        RETURN d.date as date, 
               c.name as concept, 
               c.type as concept_type,
               count(*) as frequency
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            return [dict(record) for record in result]


# ============================================================================
# Layout & Image Processing
# ============================================================================

def load_layout_batch(json_dir: Path, file_pattern: str = "*.json") -> Dict:
    """Load multiple layout JSON files"""
    layouts = {}
    for json_file in Path(json_dir).glob(file_pattern):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            layouts[json_file.stem] = data
    return layouts


def extract_detection_stats(layout_data: Dict) -> Dict:
    """Extract statistics from layout detection data"""
    stats = {
        'total_detections': layout_data.get('total_detections', 0),
        'by_class': layout_data.get('summary', {}).get('by_class', {}),
        'has_picture': any(d['class'] == 'Picture' for d in layout_data.get('detections', [])),
        'text_blocks': sum(1 for d in layout_data.get('detections', []) if d['class'] == 'Text'),
        'avg_confidence': np.mean([d['confidence'] for d in layout_data.get('detections', [])])
    }
    return stats


def draw_layout_overlays(
    image_path: str, 
    detections: List[Dict],
    selected_classes: List[str] = None,
    show_labels: bool = True,
    alpha: int = 80
) -> Image:
    """Draw bounding boxes on newspaper image"""
    
    # Color scheme
    colors = {
        'Text': (59, 130, 246),      # Blue
        'Picture': (16, 185, 129),   # Green
        'Caption': (245, 158, 11),   # Orange
        'Section-header': (139, 92, 246)  # Purple
    }
    
    img = Image.open(image_path).convert('RGBA')
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    for det in detections:
        cls = det['class']
        
        # Filter by selected classes
        if selected_classes and cls not in selected_classes:
            continue
        
        bbox = det['bbox']
        color = colors.get(cls, (128, 128, 128))
        
        # Draw filled rectangle
        draw.rectangle(
            [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
            outline=color + (255,),
            fill=color + (alpha,),
            width=3
        )
        
        # Draw label
        if show_labels:
            label = f"{cls} ({det['confidence']:.2f})"
            # Try to load a font, fall back to default
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Text background
            text_bbox = draw.textbbox((bbox['x1'], bbox['y1'] - 20), label, font=font)
            draw.rectangle(text_bbox, fill=color + (200,))
            draw.text((bbox['x1'], bbox['y1'] - 20), label, fill='white', font=font)
    
    # Composite overlay onto image
    result = Image.alpha_composite(img, overlay)
    return result.convert('RGB')


def extract_text_from_region(layout_data: Dict, bbox: Dict) -> str:
    """Extract text from a specific region"""
    texts = []
    for det in layout_data.get('detections', []):
        det_bbox = det['bbox']
        
        # Check if bboxes overlap
        if (det_bbox['x1'] < bbox['x2'] and det_bbox['x2'] > bbox['x1'] and
            det_bbox['y1'] < bbox['y2'] and det_bbox['y2'] > bbox['y1']):
            
            if det.get('text_content'):
                texts.append(det['text_content'])
    
    return '\n\n'.join(texts)


# ============================================================================
# Emotion & NER Processing
# ============================================================================

def load_emotion_data(emotion_csv: str) -> pd.DataFrame:
    """Load emotion classification results"""
    df = pd.read_csv(emotion_csv)
    # Expected columns: date, page_id, emotion, confidence, sadness, joy, anger, fear, etc.
    return df


def aggregate_emotions_by_date(emotion_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate emotions by date"""
    return emotion_df.groupby('date').agg({
        'sadness': 'mean',
        'joy': 'mean',
        'anger': 'mean',
        'fear': 'mean',
        'surprise': 'mean',
        'neutral': 'mean'
    }).reset_index()


def load_ner_data(ner_json_dir: Path) -> pd.DataFrame:
    """Load NER results from multiple JSON files"""
    all_entities = []
    
    for json_file in Path(ner_json_dir).glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Extract date from filename or data
            date = data.get('date') or json_file.stem.split('_')[1]
            
            for entity in data.get('entities', []):
                all_entities.append({
                    'date': date,
                    'entity': entity['text'],
                    'type': entity['label'],
                    'start': entity.get('start'),
                    'end': entity.get('end'),
                    'confidence': entity.get('score', 1.0)
                })
    
    return pd.DataFrame(all_entities)


def get_top_entities(ner_df: pd.DataFrame, entity_type: str = None, top_n: int = 20) -> pd.DataFrame:
    """Get most frequent entities"""
    if entity_type:
        ner_df = ner_df[ner_df['type'] == entity_type]
    
    return ner_df.groupby(['entity', 'type']).size().reset_index(name='count').sort_values('count', ascending=False).head(top_n)


def entity_timeline(ner_df: pd.DataFrame, entity_name: str) -> pd.DataFrame:
    """Get timeline for specific entity"""
    entity_data = ner_df[ner_df['entity'] == entity_name]
    return entity_data.groupby('date').size().reset_index(name='mentions')


# ============================================================================
# Image-Caption Pairs
# ============================================================================

def load_image_caption_pairs(pairs_json: str) -> List[Dict]:
    """Load extracted image-caption pairs"""
    with open(pairs_json, 'r', encoding='utf-8') as f:
        return json.load(f)


def filter_pairs_by_date(pairs: List[Dict], start_date: str, end_date: str) -> List[Dict]:
    """Filter image-caption pairs by date range"""
    filtered = []
    for pair in pairs:
        # Extract date from source path (e.g., "1901/04/23/01/...")
        date_parts = pair['source_image'].split('/')[:3]
        date_str = '-'.join(date_parts)  # "1901-04-23"
        
        if start_date <= date_str <= end_date:
            filtered.append(pair)
    
    return filtered


def extract_caption_entities(pairs: List[Dict], ner_df: pd.DataFrame) -> pd.DataFrame:
    """Link captions to their entities"""
    caption_entities = []
    
    for pair in pairs:
        caption_text = pair.get('caption', {}).get('text_content', '')
        
        # Find entities mentioned in caption
        for _, entity_row in ner_df.iterrows():
            if entity_row['entity'].lower() in caption_text.lower():
                caption_entities.append({
                    'image': pair['source_image'],
                    'caption': caption_text,
                    'entity': entity_row['entity'],
                    'entity_type': entity_row['type']
                })
    
    return pd.DataFrame(caption_entities)


# ============================================================================
# Combined Analysis
# ============================================================================

def create_multimodal_index(
    layout_dir: Path,
    emotion_csv: str,
    ner_json_dir: Path,
    pairs_json: str
) -> pd.DataFrame:
    """Create unified index of all pages with metadata"""
    
    # Load all data
    layouts = load_layout_batch(layout_dir)
    emotions = load_emotion_data(emotion_csv)
    ner_data = load_ner_data(ner_json_dir)
    pairs = load_image_caption_pairs(pairs_json)
    
    index = []
    
    for page_id, layout in layouts.items():
        # Extract date from layout
        date = layout.get('date') or page_id.split('_')[1]
        
        # Get emotion for this page
        page_emotions = emotions[emotions['page_id'] == page_id]
        
        # Get entities for this page
        page_entities = ner_data[ner_data['date'] == date]
        
        # Get image pairs
        page_pairs = [p for p in pairs if page_id in p['source_image']]
        
        index.append({
            'page_id': page_id,
            'date': date,
            'image_file': layout.get('image_file'),
            'total_detections': layout.get('total_detections', 0),
            'has_picture': any(p for p in page_pairs),
            'dominant_emotion': page_emotions['emotion'].mode()[0] if not page_emotions.empty else 'unknown',
            'emotion_score': page_emotions['confidence'].mean() if not page_emotions.empty else 0,
            'entity_count': len(page_entities),
            'top_entity': page_entities['entity'].mode()[0] if not page_entities.empty else None
        })
    
    return pd.DataFrame(index)


# ============================================================================
# Export Functions
# ============================================================================

def export_search_results(results: List[Dict], output_file: str):
    """Export search results to JSON"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def export_network_graph(neo4j_data: List[Dict], output_file: str):
    """Export network data for visualization"""
    import networkx as nx
    
    G = nx.Graph()
    
    for record in neo4j_data:
        entity = record['entity']
        date = record['date']
        
        G.add_node(entity, type='entity', entity_type=record.get('entity_type'))
        G.add_node(date, type='date')
        G.add_edge(date, entity, weight=1)
    
    # Export as GEXF for Gephi or similar
    nx.write_gexf(G, output_file)
    
    return G


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Load and process data
    
    # Connect to Neo4j
    neo4j = Neo4jConnector("bolt://localhost:7687", "neo4j", "password")
    
    # Get entity network
    network = neo4j.get_entity_network("1901-01-01", "1901-12-31")
    print(f"Retrieved {len(network)} network records")
    
    # Load layout data
    layouts = load_layout_batch(Path("./data/layouts"))
    print(f"Loaded {len(layouts)} layout files")
    
    # Load emotions
    emotions = load_emotion_data("./data/emotions.csv")
    emotion_timeline = aggregate_emotions_by_date(emotions)
    print(emotion_timeline.head())
    
    # Load NER data
    ner_data = load_ner_data(Path("./data/ner"))
    top_entities = get_top_entities(ner_data, entity_type='PERSON', top_n=10)
    print(top_entities)
    
    neo4j.close()