import json
import pprint
from pathlib import Path
from typing import Dict, List, Any

def load_json_from_folder(folder_path: str) -> Dict[str, Any]:
    """
    Finds all .json files in a folder, parses them, and returns a
    dictionary mapping the filename to its parsed content.

    Args:
        folder_path: A string path to the directory.

    Returns:
        A dictionary where keys are filenames (e.g., "config.json")
        and values are the parsed JSON data (dicts, lists, etc.).
    """
    # 1. Convert the string path to a Path object for easier handling
    path = Path(folder_path)

    # 2. Check if the path is a valid directory
    if not path.is_dir():
        print(f"Error: Path '{path}' is not a valid directory.")
        return {}

    # 3. This dictionary will store our results
    parsed_data = {}

    # 4. Use .glob('*.json') to efficiently find only .json files
    for json_file in path.glob('*.json'):
        try:
            # 5. Open and read the file
            # 'with open' handles closing the file automatically
            with open(json_file, 'r', encoding='utf-8') as f:
                
                # 6. Parse the JSON data
                data = json.load(f)
                
                # 7. Store the data in our dictionary using the filename as the key
                # .name gives you just the filename (e.g., "config.json")
                parsed_data[json_file] = data

        except json.JSONDecodeError:
            # Handle files that are not valid JSON
            print(f"Warning: Skipping '{json_file.name}'. Invalid JSON format.")
        except IOError as e:
            # Handle files that can't be read (e.g., permission issues)
            print(f"Warning: Skipping '{json_file.name}'. Could not read file: {e}")
        except Exception as e:
            # Catch other potential errors
            print(f"Warning: An unexpected error occurred with '{json_file.name}': {e}")

    # 8. Return the complete dictionary
    return parsed_data

if __name__ == "__main__":
    basic_graphs = load_json_from_folder("../dataset/basic_graphs")
    for file_name, dat in basic_graphs.items():
        print(file_name)
        for func, graph in dat.items():
            print(func)
            assert "edges" in graph
            assert "nodes" in graph
            nodes = []
            reverse_index = []
            for node in graph["nodes"]:
                print(node)
            for edge in graph["edges"]:
                print(edge)
            