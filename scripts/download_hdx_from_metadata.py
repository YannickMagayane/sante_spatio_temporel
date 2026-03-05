import os
import csv
import requests
import logging
from urllib.parse import urlparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, output_dir):
    try:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_file"
        
        output_path = os.path.join(output_dir, filename)
        
        if os.path.exists(output_path):
            logger.info(f"File already exists, skipping: {filename}")
            return output_path
            
        logger.info(f"Downloading {url} to {output_path}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Successfully downloaded {filename}")
        return output_path
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return None

def process_metadata_directory(metadata_dir="data/hdx_cvc", output_dir="data/raw_hdx"):
    os.makedirs(output_dir, exist_ok=True)
    
    total_downloaded = 0
    total_urls_found = 0
    
    if not os.path.exists(metadata_dir):
        logger.error(f"Metadata directory {metadata_dir} does not exist.")
        return
        
    for filename in os.listdir(metadata_dir):
        if not filename.endswith('.csv'):
            continue
            
        filepath = os.path.join(metadata_dir, filename)
        logger.info(f"Processing metadata file: {filename}")
        
        # Read the csv file and extract download URLs
        urls_to_download = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    field = row.get('Field', '')
                    if field.endswith('_download_url'):
                        url = row.get('Value')
                        if url:
                            urls_to_download.append(url)
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
            continue
            
        total_urls_found += len(urls_to_download)
        for url in urls_to_download:
            if download_file(url, output_dir):
                total_downloaded += 1

    logger.info(f"Finished. Downloaded {total_downloaded}/{total_urls_found} files from metadata links.")

if __name__ == "__main__":
    process_metadata_directory()
