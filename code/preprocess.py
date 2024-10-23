import os
import json
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
import argparse
import logging
from tqdm import tqdm
import multiprocessing
from functools import partial
import random

def setup_logging(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)



import os
import json
import logging

def load_metadata(metadata_dir):
    for subfolder in os.listdir(metadata_dir):
        subfolder_path = os.path.join(metadata_dir, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.json'):
                    file_path = os.path.join(subfolder_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                line = line.strip()
                                if line:  # Skip empty lines
                                    try:
                                        yield json.loads(line)
                                    except json.JSONDecodeError as e:
                                        logging.error(f"Error decoding JSON object in file {file_path}: {str(e)}")
                    except UnicodeDecodeError as e:
                        logging.error(f"Error reading file {file_path}: {str(e)}")
                        logging.info(f"Attempting to read {file_path} with 'latin-1' encoding")
                        try:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                for line in f:
                                    line = line.strip()
                                    if line:  # Skip empty lines
                                        try:
                                            yield json.loads(line)
                                        except json.JSONDecodeError as e:
                                            logging.error(f"Error decoding JSON object in file {file_path}: {str(e)}")
                        except Exception as e:
                            logging.error(f"Failed to read {file_path} with 'latin-1' encoding: {str(e)}")
import logging
from collections import Counter
from tqdm import tqdm

def process_tags(metadata_generator, sample_rate):
    tag_counts = Counter()
    processed_metadata = []
    total_items = 0
    processed_items = 0
    
    for i, item in enumerate(tqdm(metadata_generator, desc="Processing metadata")):
        total_items += 1
        if i % sample_rate == 0:  # Only process every nth item
            if 'tags' in item and isinstance(item['tags'], list):
                tags = [tag['name'] for tag in item['tags'] if isinstance(tag, dict) and 'name' in tag]
                tag_counts.update(tags)
                item['tags'] = tags
                processed_metadata.append(item)
                processed_items += 1
            else:
                logging.warning(f"Skipping item {i} due to invalid 'tags' field")
    
    logging.info(f"Total items in metadata: {total_items}")
    logging.info(f"Processed items: {processed_items}")
    
    return processed_metadata, tag_counts
def filter_tags(tag_counts, min_occurrences=100):
    return [tag for tag, count in tag_counts.items() if count >= min_occurrences]

def create_tag_mapping(filtered_tags):
    return {tag: i for i, tag in enumerate(filtered_tags)}

def process_image(metadata, tag_mapping, image_dir, output_dir):
    image_id = metadata['id']
    folder = f"0{image_id[-3:]}"  # Assuming the last 3 digits of the ID determine the subfolder
    image_path = os.path.join(image_dir, folder, f"{image_id}.jpg")
    
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            if width != 512 or height != 512:
                logging.warning(f"Image {image_path} is not 512x512. Actual size: {width}x{height}")
        
        tags = metadata['tags']
        encoded_tags = [tag for tag in tags if tag in tag_mapping]
        
        processed_image = {
            'id': image_id,
            'tags': encoded_tags,
            'file_path': image_path
        }
        
        json_path = os.path.join(output_dir, f"{image_id}.json")
        with open(json_path, 'w') as f:
            json.dump(processed_image, f)
        
        return processed_image
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
    except Image.UnidentifiedImageError:
        logging.error(f"Cannot identify image file: {image_path}")
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
    return None

import os
import json
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter
import multiprocessing
from functools import partial

def main(args):
    setup_logging(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Starting preprocessing with the following arguments:")
    logging.info(f"Image directory: {args.image_dir}")
    logging.info(f"Metadata directory: {args.metadata_dir}")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Sample rate: {args.sample_rate}")
    logging.info(f"Minimum tag occurrences: {args.min_tag_occurrences}")

    # Load and process metadata
    metadata_generator = load_metadata(args.metadata_dir)
    processed_metadata, tag_counts = process_tags(metadata_generator, args.sample_rate)
    logging.info(f"Loaded metadata for {len(processed_metadata)} images (1 in {args.sample_rate})")
    logging.info(f"Total unique tags: {len(tag_counts)}")

    if len(processed_metadata) == 0:
        logging.error("No metadata was processed. Please check your metadata directory and file permissions.")
        return

    # Filter tags and create mapping
    filtered_tags = filter_tags(tag_counts, args.min_tag_occurrences)
    logging.info(f"Filtered tags: {len(filtered_tags)}")
    tag_mapping = create_tag_mapping(filtered_tags)

    # Process images in parallel
    num_processes = multiprocessing.cpu_count()
    logging.info(f"Using {num_processes} processes for parallel processing")
    with multiprocessing.Pool(processes=num_processes) as pool:
        process_image_partial = partial(process_image, 
                                        tag_mapping=tag_mapping, 
                                        image_dir=args.image_dir, 
                                        output_dir=args.output_dir)
        
        processed_data = list(tqdm(
            pool.imap(process_image_partial, processed_metadata),
            total=len(processed_metadata),
            desc="Processing images"
        ))

    # Remove None values (failed processing)
    processed_data = [item for item in processed_data if item is not None]
    logging.info(f"Total processed images: {len(processed_data)}")

    if len(processed_data) == 0:
        logging.error("No images were successfully processed. Please check your image directory and file permissions.")
        return

    # Create train/val/test splits
    try:
        train_val, test = train_test_split(processed_data, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.2, random_state=42)
    except ValueError as e:
        logging.error(f"Error during train/test split: {str(e)}")
        return

    # Save split indices
    for name, data in [('train', train), ('val', val), ('test', test)]:
        with open(os.path.join(args.output_dir, f'{name}_index.json'), 'w') as f:
            json.dump([item['id'] for item in data], f)

    # Save the tag mapping
    with open(os.path.join(args.output_dir, 'tag_mapping.json'), 'w') as f:
        json.dump(tag_mapping, f)

    logging.info("Preprocessing completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess anime illustration dataset for multi-label classification")
    parser.add_argument("--image_dir", default=r"danbooru-images\danbooru-images", help="Directory containing subfolders with image files")
    parser.add_argument("--metadata_dir", default="danbooru-metadata", help="Directory containing subfolders with metadata JSON files")
    parser.add_argument("--output_dir", default="processed_data", help="Directory to save processed data")
    parser.add_argument("--log_file", default="preprocessing.log", help="File to save log messages")
    parser.add_argument("--min_tag_occurrences", type=int, default=100, help="Minimum number of occurrences for a tag to be included")
    parser.add_argument("--sample_rate", type=int, default=4, help="Process 1 in every n images")
    args = parser.parse_args()
    
    main(args)
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Preprocess anime illustration dataset for multi-label classification")
#     parser.add_argument("--image_dir", default="danbooru-images/danbooru-images/", help="Directory containing image files")
#     parser.add_argument("--metadata_dir", default="danbooru-metadata/danbooru-metadata/", help="Directory containing metadata JSON files")
#     parser.add_argument("--output_dir", default="processed_data", help="Directory to save processed data")
#     parser.add_argument("--log_file", default="preprocessing.log", help="File to save log messages")
#     parser.add_argument("--min_tag_occurrences", type=int, default=100, help="Minimum number of occurrences for a tag to be included")
#     parser.add_argument("--sample_rate", type=int, default=10, help="Process 1 in every n images")
#     args = parser.parse_args()
    
#     main(args)