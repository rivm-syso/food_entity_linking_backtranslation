# OpenAI embedder (text-embedding-ada-002)
import os
import json
from pathlib import Path
import numpy as np
from azure.openai import AzureOpenAI  # Assuming this is correct import

class Embedder:
    """
    OpenAI embedder using Azure OpenAI (text-embedding-ada-002).
    """

    def __init__(self):
        self.create_client()
        
    def get_model_config(self, config):
        azure_endpoint=config['AZURE_ENDPOINT']
        api_key=config['PUBLIC_OPENAI_API_KEY']
        api_version=config['LOCAL_OPENAI_API_VERSION']
        model_name=config['LOCAL_OPENAI_DEPLOYMENT_EMBEDDINGS']
        return azure_endpoint, api_key, api_version, model_name
    
    def create_client(self):
        # Read config
        # Import the LLM configurations
        current_directory = os.getcwd()
        parent_directory = Path(current_directory).parent
        config_file = os.path.join(parent_directory, 'llm_config.json')

        if os.path.exists(config_file):
            with open(config_file) as file:
                config = json.load(file)
        
        # Get configurations
        azure_endpoint, api_key, api_version, model_name = self.get_model_config(config)

        self.model_name = model_name
        
        # Create client
        self.client = AzureOpenAI(
                    azure_endpoint=azure_endpoint,
                    api_key=api_key,
                    api_version=api_version
                )
    
    def create_embedding(self, text):
        # Create embeddings
        response = self.client.embeddings.create(
                input=text,
                model=self.model_name
            ).data[0].embedding
    
        return np.vstack(response).transpose()


# Function to get parent IDs from a food item in foodon
# If there is no parent, returns None
def get_parent_ids(df_foodon, foodon_id):
    parent_row = df_foodon[df_foodon["Class ID"] == foodon_id].reset_index()["Parents"]
    if len(parent_row) == 0:
        return None
    else:
        relationships = parent_row[0]
    parent_ids = relationships.split("|")
    return parent_ids

# Function to get all parent IDs from a food item in foodon
# If there is no parent, returns empty list
def get_all_parent_ids(df_foodon, foodon_id, all_ids = []):
    all_ids.append(foodon_id)
    direct_parent_ids = get_parent_ids(df_foodon, foodon_id)
    
    if direct_parent_ids is not None:
        for direct_parent_id in direct_parent_ids:
            if direct_parent_id is not None:
                parent_ids = get_all_parent_ids(df_foodon, direct_parent_id, all_ids)
    return all_ids

def get_children_ids(df_foodon, foodon_id):
    parent_list = df_foodon['Parents'].fillna("").values
    children_indices = [i for i, parent in enumerate(parent_list) if foodon_id in parent.split("|")]
    children_ids = df_foodon['Class ID'].iloc[children_indices]
    return list(children_ids)

# Function to get all children IDs from a food item in foodon
# If there are no children, returns empty list
def get_all_children_ids(df_foodon, foodon_id, all_ids = []):
    if foodon_id not in all_ids:
        all_ids.append(foodon_id)
        direct_children_ids = get_children_ids(df_foodon, foodon_id)

        if direct_children_ids is not None:
            for direct_children_id in direct_children_ids:
                if direct_children_id is not None:
                    children_ids = get_all_children_ids(df_foodon, direct_children_id, all_ids)
    return all_ids
