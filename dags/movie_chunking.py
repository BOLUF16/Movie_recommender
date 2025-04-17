from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from kaggle.api.kaggle_api_extended import KaggleApi
from azure.storage.blob import BlobServiceClient
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from datetime import datetime, timedelta
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
KAGGLE_DATASET_NAME = "tmdb-movies-dataset-2023-930k-movies"
KAGGLE_DATASET_OWNER = "asaniczka"
TEMP_DIR = "tmp/airflow_data_path"
AZURE_CONTAINER_NAME = "moviedata"
CSV_FILENAME = "tmdb_movies_data.csv"

# Default arguments for the DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 3, 8),
    "retries": 3,
    "retry_delay": timedelta(minutes=5)
}

def download_kaggle_data(ti):
    """
    Downloads a dataset from Kaggle and stores it in a temporary directory.

    Args:
        ti (TaskInstance): The Airflow TaskInstance object used to push data to XCom.

    Steps:
        1. Retrieves Kaggle credentials from Airflow Variables.
        2. Authenticates with the Kaggle API.
        3. Downloads and unzips the dataset to a temporary directory.
        4. Pushes the path of the downloaded file to XCom for downstream tasks.

    Raises:
        Exception: If there is an error during the download process.
    """
    try:
        # Retrieve Kaggle credentials from Airflow Variables
        kaggle_credential = Variable.get("kaggle_auth", deserialize_json=True)
        username = kaggle_credential["username"]
        key = kaggle_credential["key"]
        
        # Set Kaggle credentials in environment variables
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        
        # Authenticate with Kaggle API
        api = KaggleApi()
        api.authenticate()

        # Create temporary directory if it doesn't exist
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)
            logger.info(f"Created temporary directory: {TEMP_DIR}")
        
        # Download and unzip the dataset
        api.dataset_download_files(
            dataset=f"{KAGGLE_DATASET_OWNER}/{KAGGLE_DATASET_NAME}",
            path=TEMP_DIR,
            unzip=True
        )
        logger.info(f"Downloaded dataset: {KAGGLE_DATASET_NAME}")

        # Push the downloaded file path to XCom
        downloaded_path = os.path.join(TEMP_DIR, CSV_FILENAME)
        ti.xcom_push(key="downloaded_path", value=downloaded_path)
        logger.info(f"Pushed downloaded path to XCom: {downloaded_path}")

    except Exception as e:
        logger.error(f"Error downloading Kaggle dataset: {str(e)}")
        raise

def check_transform_upload(ti):
    """
    Checks if data exists in Azure Blob Storage, transforms the data, and uploads it if necessary.

    Args:
        ti (TaskInstance): The Airflow TaskInstance object used to pull and push data to XCom.

    Steps:
        1. Retrieves the Azure connection string from Airflow Variables.
        2. Checks if the dataset already exists in Azure Blob Storage.
        3. If the dataset does not exist, processes the downloaded data and uploads it to Azure.
        4. If the dataset exists, compares the new data with the existing data and uploads only the new entries.
        5. Pushes the path of the combined data to XCom for downstream tasks.

    Raises:
        Exception: If there is an error during the transformation or upload process.
    """
    try:
        # Retrieve Azure connection string from Airflow Variables
        connection_string = Variable.get("Azure_connection_string")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
        
        # Define Azure blob name and local paths
        azure_blob_name = f"{KAGGLE_DATASET_NAME}.csv"
        blob_client = container_client.get_blob_client(azure_blob_name)
        combined_data_path = os.path.join(TEMP_DIR, "combined_data.csv")
        
        # Check if the blob exists in Azure
        if not blob_client.exists():
            logger.info("Blob does not exist in Azure. Processing new data.")
            
            # Retrieve downloaded data path from XCom
            downloaded_path = ti.xcom_pull(task_ids="download_data", key="downloaded_path")
            transformed_data_path = os.path.join(TEMP_DIR, "transformed_data.csv")
            all_transformed_data = pd.DataFrame()
            
            # Process data in chunks
            for chunk in pd.read_csv(downloaded_path, chunksize=10000):
                # Filter and transform data
                data = chunk[chunk["status"] == "Released"]
                data = data[['title', 'vote_average', 'release_date',
                            'original_language', 'original_title', 'overview', 'genres',
                            'production_companies', 'production_countries']]
                
                # Combine movie attributes into a single column
                data["Combined_movie_attr"] = data.apply(
                    lambda row: f"Title: {row['title']}\nOverview: {row['overview']}\nGenres: {row['genres']}\n"
                               f"Vote Average: {row['vote_average']}\nRelease Date: {row['release_date']}\n"
                               f"Language: {row['original_language']}\nOriginal Title: {row['original_title']}\n"
                               f"Production Companies: {row['production_companies']}\n"
                               f"Production Countries: {row['production_countries']}", 
                    axis=1
                )
                
                all_transformed_data = pd.concat([all_transformed_data, data], ignore_index=True)
            
            # Save transformed data to CSV
            all_transformed_data.to_csv(transformed_data_path, index=False)
            all_transformed_data.to_csv(combined_data_path, index=False) 
            logger.info(f"Saved transformed data to: {transformed_data_path}")
            
            # Upload transformed data to Azure
            with open(transformed_data_path, "rb") as data_file:
                container_client.upload_blob(name=azure_blob_name, data=data_file, overwrite=True)
            logger.info(f"Uploaded data to Azure blob: {azure_blob_name}")
            
            # Push combined data path to XCom
            ti.xcom_push(key="check_transform", value=combined_data_path)
        
        else:
            logger.info("Blob exists in Azure. Checking for new data.")
            
            # Download existing Azure data
            azure_data_path = os.path.join(TEMP_DIR, "azure_movie_data.csv")
            with open(azure_data_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            logger.info(f"Downloaded existing Azure data to: {azure_data_path}")
            
            # Load existing Azure data
            azure_data = pd.read_csv(azure_data_path)
            
            # Process new downloaded data
            downloaded_path = ti.xcom_pull(task_ids="download_data", key="downloaded_path")
            transformed_data_path = os.path.join(TEMP_DIR, "transformed_data.csv")
            all_transformed_data = pd.DataFrame()
            
            for chunk in pd.read_csv(downloaded_path, chunksize=10000):
                data = chunk[chunk["status"] == "Released"]
                data = data[['title', 'vote_average', 'release_date',
                            'original_language', 'original_title', 'overview', 'genres',
                            'production_companies', 'production_countries']]
                
                data["Combined_movie_attr"] = data.apply(
                    lambda row: f"Title: {row['title']}\nOverview: {row['overview']}\nGenres: {row['genres']}\n"
                               f"Vote Average: {row['vote_average']}\nRelease Date: {row['release_date']}\n"
                               f"Language: {row['original_language']}\nOriginal Title: {row['original_title']}\n"
                               f"Production Companies: {row['production_companies']}\n"
                               f"Production Countries: {row['production_countries']}", 
                    axis=1
                )
                
                all_transformed_data = pd.concat([all_transformed_data, data], ignore_index=True)
            
            # Save transformed data
            all_transformed_data.to_csv(transformed_data_path, index=False)
            logger.info(f"Saved transformed data to: {transformed_data_path}")
            
            # Find new data (not in Azure)
            new_data = all_transformed_data[~all_transformed_data["title"].isin(azure_data["title"])]
            
            if len(new_data) == 0:
                logger.info("No new data found.")
                ti.xcom_push(key="check_transform", value="No new data found")
                return
            
            # Combine existing and new data
            combined_data = pd.concat([azure_data, new_data], ignore_index=True)
            combined_data.to_csv(combined_data_path, index=False)
            logger.info(f"Combined new and existing data: {combined_data_path}")
            
            # Upload updated data to Azure
            with open(combined_data_path, "rb") as data_file:
                container_client.upload_blob(name=azure_blob_name, data=data_file, overwrite=True)
            logger.info(f"Uploaded updated data to Azure blob: {azure_blob_name}")
            
            # Push combined data path to XCom
            ti.xcom_push(key="check_transform", value=combined_data_path)
    
    except Exception as e:
        logger.error(f"Error in check_transform_upload: {str(e)}")
        raise

def chunking_data_qdrant(ti):
    """
    Embeds document chunks into a Qdrant vector database.

    Args:
        ti (TaskInstance): The Airflow TaskInstance object used to pull data from XCom.

    Steps:
        1. Retrieves Qdrant and HuggingFace API keys from Airflow Variables.
        2. Initializes the Qdrant client and checks if the collection exists.
        3. Retrieves the combined data path from XCom.
        4. Processes the data in chunks, generates embeddings, and uploads them to Qdrant.
        5. Logs the progress and any errors encountered.

    Raises:
        Exception: If there is an error during the embedding or upload process.
    """
    try:
        # Retrieve Qdrant and HuggingFace API keys from Airflow Variables
        qdrant_key = Variable.get("qdrant_api_key")
        huggingface_api_key = Variable.get("huggingface_api_key")
        
        # Initialize Qdrant client
        client = QdrantClient(
            url="https://398e383f-3b1b-4900-b002-88776d6c621f.us-east4-0.gcp.cloud.qdrant.io:6333", 
            api_key=qdrant_key
        )
        
        collection_name = "movies"
        
        # Check if collection exists, create if it doesn't
        try:
            collections = client.get_collections()
            collection_exists = collection_name in [col.name for col in collections.collections]
            
            if not collection_exists:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            elif collection_exists:
                logger.info(f"Using existing Qdrant collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error with Qdrant collection: {str(e)}")
            raise
        
        # Get the file path from XCom
        process_data_path = ti.xcom_pull(task_ids="process_and_upload", key="check_transform")
        
        # If no new data was found, exit early
        if process_data_path == "No new data found":
            logger.info("No new data to embed.")
            return
        
        # Get the actual combined data path
        combined_data_path = os.path.join(TEMP_DIR, "combined_data.csv")
        
        # Initialize HuggingFace embeddings
        embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=huggingface_api_key, 
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        # Process data in chunks
        chunk_id = 0
        
        for chunk_idx, chunk in enumerate(pd.read_csv(combined_data_path, chunksize=10000)):
            logger.info(f"Processing chunk {chunk_idx+1}")
            
            # Get the text data for embedding
            text_data = chunk["Combined_movie_attr"].tolist()
            
            # Get embeddings for all texts in the chunk at once
            try:
                embeddings = embedding_model.embed_documents(text_data)
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise
            
            points = []
            
            # Create points with the embeddings and metadata
            for i, (_, row) in enumerate(chunk.iterrows()):
                try:
                    points.append(
                        models.PointStruct(
                            id=chunk_id + i,
                            vector=embeddings[i],
                            payload={
                                "title": row['title'],
                                "vote_average": row['vote_average'],
                                "release_date": row['release_date'],
                                "original_language": row['original_language'],
                                "original_title": row['original_title'],
                                "overview": row['overview'],
                                "genres": row['genres'],
                                "combined_attr": row['Combined_movie_attr']
                            }
                        )
                    )
                except Exception as e:
                    logger.error(f"Error creating point for record {i}: {str(e)}")
                    continue
            
            # Update Qdrant
            if points:
                try:
                    client.upsert(
                        collection_name=collection_name,
                        points=points
                    )
                    logger.info(f"Successfully upserted {len(points)} points")
                except Exception as e:
                    logger.error(f"Error upserting points: {str(e)}")
                    raise
            
            # Update the ID offset for the next chunk
            chunk_id += len(chunk)
    
    except Exception as e:
        logger.error(f"Error in chunking_data_qdrant: {str(e)}")
        raise

# Create the DAG
with DAG(
    'movie_data_processing',
    default_args=default_args,
    description='A DAG to download, transform, upload, and embed movie data',
    schedule_interval="@weekly",
    catchup=False
) as dag:
    
    download_task = PythonOperator(
        task_id='download_data',
        python_callable=download_kaggle_data,
    )
    
    process_upload_task = PythonOperator(
        task_id='process_and_upload',
        python_callable=check_transform_upload,
    )
    
    embed_qdrant_task = PythonOperator(
        task_id='embed_qdrant',
        python_callable=chunking_data_qdrant,
    )
    
    # Set task dependencies
    download_task >> process_upload_task >> embed_qdrant_task