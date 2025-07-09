#!/usr/bin/env python3
"""
Script to load PrimeKG multimodal data into Milvus database.
This script runs after Milvus container is ready and loads the .pkl file data.
"""

import os
import sys
import subprocess
import pickle
import time
import logging
from typing import Dict, Any, List

def install_packages():
    """Install required packages."""
    packages = [
        "pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12",
        "pip install --extra-index-url=https://pypi.nvidia.com dask-cudf-cu12", 
        "pip install pymilvus==2.5.11",
        "pip install numpy==1.26.4",
        "pip install pandas==2.1.3",
        "pip install tqdm==4.67.1",
        "pip install torch_geometric==2.6.1",
        "pip install torch==2.2.2"
    ]
    
    print("[DATA LOADER] Installing required packages...")
    for package_cmd in packages:
        print(f"[DATA LOADER] Running: {package_cmd}")
        result = subprocess.run(package_cmd.split(), capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[DATA LOADER] Error installing package: {result.stderr}")
            sys.exit(1)
    print("[DATA LOADER] All packages installed successfully!")

# Install packages first
install_packages()

import cudf
import cupy as cp
import numpy as np
from pymilvus import (
    db,
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    MilvusClient
)
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='[DATA LOADER] %(message)s')
logger = logging.getLogger(__name__)

class MilvusDataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.milvus_host = config.get('milvus_host', 'localhost')
        self.milvus_port = config.get('milvus_port', '19530')
        self.milvus_user = config.get('milvus_user', 'root')
        self.milvus_password = config.get('milvus_password', 'Milvus')
        self.milvus_database = config.get('milvus_database', 't2kg_primekg')
        self.pkl_file_path = config.get('pkl_file_path', 'tests/files/biobridge_multimodal_pyg_graph.pkl')
        self.batch_size = config.get('batch_size', 500)
        
    def normalize_matrix(self, m, axis=1):
        """Normalize each row of a 2D matrix using CuPy."""
        norms = cp.linalg.norm(m, axis=axis, keepdims=True)
        return m / norms

    def normalize_vector(self, v):
        """Normalize a vector using CuPy."""
        v = cp.asarray(v)
        norm = cp.linalg.norm(v)
        return v / norm

    def connect_to_milvus(self):
        """Connect to Milvus and setup database."""
        logger.info(f"Connecting to Milvus at {self.milvus_host}:{self.milvus_port}")
        
        connections.connect(
            alias="default",
            host=self.milvus_host,
            port=self.milvus_port,
            user=self.milvus_user,
            password=self.milvus_password
        )
        
        # Check if database exists, create if it doesn't
        if self.milvus_database not in db.list_database():
            logger.info(f"Creating database: {self.milvus_database}")
            db.create_database(self.milvus_database)
        
        # Switch to the desired database
        db.using_database(self.milvus_database)
        logger.info(f"Using database: {self.milvus_database}")

    def load_graph_data(self):
        """Load the pickle file containing graph data."""
        logger.info(f"Loading graph data from: {self.pkl_file_path}")
        
        if not os.path.exists(self.pkl_file_path):
            raise FileNotFoundError(f"Pickle file not found: {self.pkl_file_path}")
        
        with open(self.pkl_file_path, 'rb') as f:
            graph = pickle.load(f)
        
        logger.info("Graph data loaded successfully")
        return graph

    def prepare_nodes_data(self, graph):
        """Prepare nodes data for Milvus insertion."""
        logger.info("Preparing nodes data...")
        
        # Normalize embeddings
        graph_desc_x_cp = cp.asarray(graph['desc_x'].tolist())
        graph_desc_x_normalized = self.normalize_matrix(graph_desc_x_cp, axis=1)
        graph_x_normalized = [self.normalize_vector(v).tolist() for v in graph['x']]

        # Create nodes DataFrame
        nodes_df = cudf.DataFrame({
            'node_id': graph['node_id'],
            'node_name': graph['node_name'],
            'node_type': graph['node_type'],
            'desc': graph['desc'],
            'desc_emb': graph_desc_x_normalized.tolist(),
            'feat': graph['enriched_node'],
            'feat_emb': graph_x_normalized,
        })
        nodes_df.reset_index(inplace=True)
        nodes_df.rename(columns={'index': 'node_index'}, inplace=True)
        
        logger.info(f"Prepared {len(nodes_df)} nodes")
        return nodes_df

    def prepare_edges_data(self, graph, nodes_df):
        """Prepare edges data for Milvus insertion."""
        logger.info("Preparing edges data...")
        
        # Normalize edge embeddings
        graph_edge_attr_cp = cp.asarray(graph['edge_attr'].tolist())
        graph_edge_attr_normalized = self.normalize_matrix(graph_edge_attr_cp, axis=1)

        # Create edges DataFrame
        edges_df = cudf.DataFrame({
            'triplet_index': graph['triplet_index'],
            'head_id': graph['head_id'],
            'head_name': graph['head_name'],
            'tail_id': graph['tail_id'],
            'tail_name': graph['tail_name'],
            'display_relation': graph['display_relation'],
            'edge_type': graph['edge_type'],
            'edge_type_str': ['|'.join(e) for e in graph['edge_type']],
            'feat': graph['enriched_edge'],
            'edge_emb': graph_edge_attr_normalized.tolist(),
        })
        
        # Merge with nodes to get indices
        edges_df = edges_df.merge(
            nodes_df[['node_index', 'node_id']],
            left_on='head_id',
            right_on='node_id',
            how='left'
        )
        edges_df.rename(columns={'node_index': 'head_index'}, inplace=True)
        edges_df.drop(columns=['node_id'], inplace=True)
        
        edges_df = edges_df.merge(
            nodes_df[['node_index', 'node_id']],
            left_on='tail_id',
            right_on='node_id',
            how='left'
        )
        edges_df.rename(columns={'node_index': 'tail_index'}, inplace=True)
        edges_df.drop(columns=['node_id'], inplace=True)
        
        logger.info(f"Prepared {len(edges_df)} edges")
        return edges_df

    def create_nodes_collection(self, nodes_df):
        """Create and populate the main nodes collection."""
        logger.info("Creating main nodes collection...")
        
        node_coll_name = f"{self.milvus_database}_nodes"
        desc_emb_dim = len(nodes_df.iloc[0]['desc_emb'].to_arrow().to_pylist()[0])
        
        node_fields = [
            FieldSchema(name="node_index", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="node_name", dtype=DataType.VARCHAR, max_length=1024,
                        enable_analyzer=True, enable_match=True),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=1024,
                        enable_analyzer=True, enable_match=True),
            FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=40960,
                        enable_analyzer=True, enable_match=True),
            FieldSchema(name="desc_emb", dtype=DataType.FLOAT_VECTOR, dim=desc_emb_dim),
        ]
        schema = CollectionSchema(fields=node_fields, description=f"Schema for collection {node_coll_name}")

        # Create collection if it doesn't exist
        if not utility.has_collection(node_coll_name):
            collection = Collection(name=node_coll_name, schema=schema)
        else:
            collection = Collection(name=node_coll_name)

        # Create indexes
        collection.create_index(field_name="node_index", index_params={"index_type": "STL_SORT"}, index_name="node_index_index")
        collection.create_index(field_name="node_name", index_params={"index_type": "INVERTED"}, index_name="node_name_index")
        collection.create_index(field_name="node_type", index_params={"index_type": "INVERTED"}, index_name="node_type_index")
        collection.create_index(field_name="desc", index_params={"index_type": "INVERTED"}, index_name="desc_index")
        collection.create_index(field_name="desc_emb", index_params={"index_type": "GPU_CAGRA", "metric_type": "IP"}, index_name="desc_emb_index")

        # Prepare and insert data
        data = [
            nodes_df["node_index"].to_arrow().to_pylist(),
            nodes_df["node_id"].to_arrow().to_pylist(),
            nodes_df["node_name"].to_arrow().to_pylist(),
            nodes_df["node_type"].to_arrow().to_pylist(),
            nodes_df["desc"].to_arrow().to_pylist(),
            cp.asarray(nodes_df["desc_emb"].list.leaves).astype(cp.float32)
                .reshape(nodes_df.shape[0], -1)
                .tolist(),
        ]

        # Insert data in batches
        total = len(data[0])
        for i in tqdm(range(0, total, self.batch_size), desc="Inserting nodes"):
            batch = [col[i:i+self.batch_size] for col in data]
            collection.insert(batch)

        collection.flush()
        logger.info(f"Nodes collection created with {collection.num_entities} entities")

    def create_node_type_collections(self, nodes_df):
        """Create separate collections for each node type."""
        logger.info("Creating node type-specific collections...")
        
        for node_type, nodes_df_ in tqdm(nodes_df.groupby('node_type'), desc="Processing node types"):
            node_coll_name = f"{self.milvus_database}_nodes_{node_type.replace('/', '_')}"
            
            desc_emb_dim = len(nodes_df_.iloc[0]['desc_emb'].to_arrow().to_pylist()[0])
            feat_emb_dim = len(nodes_df_.iloc[0]['feat_emb'].to_arrow().to_pylist()[0])
            
            node_fields = [
                FieldSchema(name="node_index", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="node_name", dtype=DataType.VARCHAR, max_length=1024,
                            enable_analyzer=True, enable_match=True),
                FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=1024,
                            enable_analyzer=True, enable_match=True),
                FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=40960,
                            enable_analyzer=True, enable_match=True),
                FieldSchema(name="desc_emb", dtype=DataType.FLOAT_VECTOR, dim=desc_emb_dim),        
                FieldSchema(name="feat", dtype=DataType.VARCHAR, max_length=40960,
                            enable_analyzer=True, enable_match=True),
                FieldSchema(name="feat_emb", dtype=DataType.FLOAT_VECTOR, dim=feat_emb_dim),
            ]
            schema = CollectionSchema(fields=node_fields, description=f"schema for collection {node_coll_name}")

            if not utility.has_collection(node_coll_name):
                collection = Collection(name=node_coll_name, schema=schema)
            else:
                collection = Collection(name=node_coll_name)

            # Create indexes
            collection.create_index(field_name="node_index", index_params={"index_type": "STL_SORT"}, index_name="node_index_index")
            collection.create_index(field_name="node_name", index_params={"index_type": "INVERTED"}, index_name="node_name_index")
            collection.create_index(field_name="node_type", index_params={"index_type": "INVERTED"}, index_name="node_type_index")
            collection.create_index(field_name="desc", index_params={"index_type": "INVERTED"}, index_name="desc_index")
            collection.create_index(field_name="desc_emb", index_params={"index_type": "GPU_CAGRA", "metric_type": "IP"}, index_name="desc_emb_index")
            collection.create_index(field_name="feat_emb", index_params={"index_type": "GPU_CAGRA", "metric_type": "IP"}, index_name="feat_emb_index")

            # Prepare data
            data = [
                nodes_df_["node_index"].to_arrow().to_pylist(),
                nodes_df_["node_id"].to_arrow().to_pylist(),
                nodes_df_["node_name"].to_arrow().to_pylist(),
                nodes_df_["node_type"].to_arrow().to_pylist(),
                nodes_df_["desc"].to_arrow().to_pylist(),
                cp.asarray(nodes_df_["desc_emb"].list.leaves).astype(cp.float32)
                    .reshape(nodes_df_.shape[0], -1)
                    .tolist(),
                nodes_df_["feat"].to_arrow().to_pylist(),
                cp.asarray(nodes_df_["feat_emb"].list.leaves).astype(cp.float32)
                    .reshape(nodes_df_.shape[0], -1)
                    .tolist(),
            ]

            # Insert data in batches
            total_rows = len(data[0])
            for i in range(0, total_rows, self.batch_size):
                batch = [col[i:i + self.batch_size] for col in data]
                collection.insert(batch)

            collection.flush()

    def create_edges_collection(self, edges_df):
        """Create and populate the edges collection."""
        logger.info("Creating edges collection...")
        
        edge_coll_name = f"{self.milvus_database}_edges"
        
        edge_fields = [
            FieldSchema(name="triplet_index", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="head_id", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="head_index", dtype=DataType.INT64),
            FieldSchema(name="tail_id", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="tail_index", dtype=DataType.INT64),
            FieldSchema(name="edge_type", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="display_relation", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="feat", dtype=DataType.VARCHAR, max_length=40960),
            FieldSchema(name="feat_emb", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        edge_schema = CollectionSchema(fields=edge_fields, description="Schema for edges collection")

        if not utility.has_collection(edge_coll_name):
            collection = Collection(name=edge_coll_name, schema=edge_schema)
        else:
            collection = Collection(name=edge_coll_name)

        # Create indexes
        collection.create_index(field_name="triplet_index", index_params={"index_type": "STL_SORT"}, index_name="triplet_index_index")
        collection.create_index(field_name="head_index", index_params={"index_type": "STL_SORT"}, index_name="head_index_index")
        collection.create_index(field_name="tail_index", index_params={"index_type": "STL_SORT"}, index_name="tail_index_index")
        collection.create_index(field_name="feat_emb", index_params={"index_type": "GPU_CAGRA", "metric_type": "IP"}, index_name="feat_emb_index")

        # Prepare data
        data = [
            edges_df["triplet_index"].to_arrow().to_pylist(),
            edges_df["head_id"].to_arrow().to_pylist(),
            edges_df["head_index"].to_arrow().to_pylist(),
            edges_df["tail_id"].to_arrow().to_pylist(),
            edges_df["tail_index"].to_arrow().to_pylist(),
            edges_df["edge_type_str"].to_arrow().to_pylist(),
            edges_df["display_relation"].to_arrow().to_pylist(),
            edges_df["feat"].to_arrow().to_pylist(),
            cp.asarray(edges_df["edge_emb"].list.leaves).astype(cp.float32)
                .reshape(edges_df.shape[0], -1)
                .tolist(),
        ]

        # Insert data in batches
        total = len(data[0])
        for i in tqdm(range(0, total, self.batch_size), desc="Inserting edges"):
            batch_data = [d[i:i+self.batch_size] for d in data]
            collection.insert(batch_data)

        collection.flush()
        logger.info(f"Edges collection created with {collection.num_entities} entities")

    def run(self):
        """Main execution method."""
        try:
            logger.info("Starting Milvus data loading process...")
            
            # Connect to Milvus
            self.connect_to_milvus()
            
            # Load graph data
            graph = self.load_graph_data()
            
            # Prepare data
            nodes_df = self.prepare_nodes_data(graph)
            edges_df = self.prepare_edges_data(graph, nodes_df)
            
            # Create collections and load data
            self.create_nodes_collection(nodes_df)
            self.create_node_type_collections(nodes_df)
            self.create_edges_collection(edges_df)
            
            # List all collections for verification
            logger.info("Data loading completed successfully!")
            logger.info("Created collections:")
            for coll in utility.list_collections():
                collection = Collection(name=coll)
                logger.info(f"  {coll}: {collection.num_entities} entities")
                
        except Exception as e:
            logger.error(f"Error during data loading: {str(e)}")
            raise


def main():
    """Main function to run the data loader."""
    # Configuration
    config = {
        'milvus_host': os.getenv('MILVUS_HOST', 'localhost'),
        'milvus_port': os.getenv('MILVUS_PORT', '19530'),
        'milvus_user': os.getenv('MILVUS_USER', 'root'),
        'milvus_password': os.getenv('MILVUS_PASSWORD', 'Milvus'),
        'milvus_database': os.getenv('MILVUS_DATABASE', 't2kg_primekg'),
        'pkl_file_path': os.getenv('PKL_FILE_PATH', 'tests/files/biobridge_multimodal_pyg_graph.pkl'),
        'batch_size': int(os.getenv('BATCH_SIZE', '500')),
    }
    
    # Create and run data loader
    loader = MilvusDataLoader(config)
    loader.run()


if __name__ == "__main__":
    main()