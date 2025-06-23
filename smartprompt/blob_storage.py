import os
from typing import List, Dict, Any, Optional
import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

STORAGE_CONNECTION_STRING_ENV_VAR = "SMARTPROMPT_STORAGE_CONNECTION_STRING"

class BlobStorageClient:
    """
    A simplified helper class for interacting with Azure Blob Storage.
    Provides functionality for creating folders, listing files, uploading, downloading,
    and deleting operations within a specific container and working folder.
    """

    def __init__(self, container_name: str, working_folder: str = ""):
        """
        Initialize the Azure Blob Storage client.
        
        Args:
            container_name: The name of the Azure Storage container (required)
            working_folder: The working folder path within the container (optional, defaults to root)
        """
        if not container_name or not container_name.strip():
            raise ValueError("Container name is required and cannot be empty")
        
        self._container_name = container_name.strip()
        self._working_folder = self._normalize_path(working_folder)
        
        # Validate connection string exists
        connection_string = os.getenv(STORAGE_CONNECTION_STRING_ENV_VAR)
        if not connection_string:
            raise ValueError(f"Storage connection string not found in environment variable: {STORAGE_CONNECTION_STRING_ENV_VAR}")
        
        # Initialize clients
        try:
            self._blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            self._container_client = self._blob_service_client.get_container_client(self._container_name)
        except Exception as e:
            raise ValueError(f"Failed to initialize blob storage client: {str(e)}")

    def _normalize_path(self, path: str) -> str:
        """Normalize a path to ensure it ends with '/' if not empty."""
        if not path:
            return ""
        normalized = path.strip().replace('\\', '/')
        return f"{normalized}/" if not normalized.endswith('/') else normalized

    def _get_full_path(self, filename: str) -> str:
        """Get the full blob path combining working folder and filename."""
        return f"{self._working_folder}{filename}"

    def create_folder(self) -> bool:
        """
        Create a virtual "folder" in Azure Blob Storage.
        
        In Azure Storage, folders are virtual and represented by prefixes in blob names.
        This method creates an empty blob with the folder name as a prefix to establish a folder structure.
        
        Returns:
            True if the folder was created or already exists, False on error
        """
        if not self._working_folder:
            # Root folder always exists
            return True
            
        try:
            blob_client = self._container_client.get_blob_client(self._working_folder)
            blob_client.upload_blob(data="", overwrite=True)
            logging.info(f"Created folder: {self._working_folder} in container {self._container_name}")
            return True
        except ResourceExistsError:
            logging.debug(f"Folder {self._working_folder} already exists in container {self._container_name}")
            return True
        except Exception as e:
            logging.error(f"Error creating folder {self._working_folder}: {str(e)}")
            return False
    
    def list_files(self) -> List[Dict[str, Any]]:
        """
        List all files in the working folder.
        
        Returns:
            A list of dictionaries containing file information
        """
        try:
            blobs = self._container_client.list_blobs(name_starts_with=self._working_folder)
            
            result = []
            for blob in blobs:
                # Skip the folder marker itself (empty blob with just the folder name)
                if blob.name == self._working_folder:
                    continue
                    
                result.append({
                    "name": blob.name,
                    "filename": os.path.basename(blob.name),
                    "size": blob.size,
                    "created_at": blob.creation_time,
                    "modified_at": blob.last_modified,
                    "content_type": blob.content_settings.content_type,
                })
            
            return result
        except Exception as e:
            logging.error(f"Error listing files in folder {self._working_folder}: {str(e)}")
            return []
    
    def upload_file(self, file_name: str, content: bytes) -> bool:
        """
        Upload a file to the working folder in blob storage.
        
        Args:
            file_name: Name of the file to upload
            content: The content to upload as bytes
            
        Returns:
            True if upload was successful, False otherwise
        """
        if not file_name or not file_name.strip():
            logging.error("File name is required and cannot be empty")
            return False
            
        try:
            full_path = self._get_full_path(file_name.strip())
            blob_client = self._container_client.get_blob_client(full_path)
            blob_client.upload_blob(content, overwrite=True)
            logging.info(f"Successfully uploaded file: {full_path}")
            return True
        except Exception as e:
            logging.error(f"Error uploading file {file_name}: {str(e)}")
            return False
    
    def download_file(self, file_name: str) -> Optional[bytes]:
        """
        Download a file from the working folder in blob storage.
        
        Args:
            file_name: Name of the file to download
            
        Returns:
            The file content as bytes, or None if file not found or error occurred
        """
        if not file_name or not file_name.strip():
            logging.error("File name is required and cannot be empty")
            return None

        try:
            full_path = self._get_full_path(file_name.strip())
            blob_client = self._container_client.get_blob_client(full_path)
            blob_data = blob_client.download_blob()
            return blob_data.readall()
        except ResourceNotFoundError:
            logging.warning(f"File not found: {file_name}")
            return None
        except Exception as e:
            logging.error(f"Error downloading file {file_name}: {str(e)}")
            return None

    def file_exists(self, file_name: str) -> bool:
        """
        Check if a file exists in the working folder.
        
        Args:
            file_name: The name of the file to check
            
        Returns:
            True if the file exists, False otherwise
        """
        if not file_name or not file_name.strip():
            return False
            
        try:
            full_path = self._get_full_path(file_name.strip())
            blob_client = self._container_client.get_blob_client(full_path)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception as e:
            logging.error(f"Error checking if file exists {file_name}: {str(e)}")
            return False

    @property
    def working_folder(self) -> str:
        """Get the current working folder path."""
        return self._working_folder

    @property
    def container_name(self) -> str:
        """Get the container name."""
        return self._container_name
