
import json
import re
from typing import Dict, Any, List

def parse_line(line: str) -> Dict[str, Any]:
    # Define regex patterns for each operation
    regex_patterns = {
        'begin': r"adding node (\d+) at level (\d+) with clock (\d+)",
        'info': r"entry path is (\S+)|L(\d+) considering forward links for (natural|concurrent) nodes (\S+) -> (\S+)",
        'finish': r"finished adding node (\d+) at clock (\d+)",
        'link_back': r"L(\d+) adding backlinks for \[(.*?)\] -> (\d+)",
        'link_forwards': r"L(\d+) inserted (\d+) -> \[(.*?)\] as forward \w+ links",
        'update_entry': r"updated entry to NodeAtLevel\(level=(\d+), node=(\d+)\)",
        'unlink': r"removed least diverse neighbor (\d+) from (\d+)",
    }
    
    # Extract the thread id from the start of the line
    thread_id, message = line.split(' ', 1)
    thread_id = int(thread_id)

    for op, pattern in regex_patterns.items():
        match = re.search(pattern, message)
        if match:
            # If the message matches a pattern, extract the relevant data
            data = match.groups()

            if op == 'begin':
                return {
                    'op': op,
                    'thread_id': thread_id,
                    'started_at': int(data[2]),
                    'node_id': int(data[0]),
                    'level': int(data[1]),
                }
            elif op == 'info':
                return {
                    'op': op,
                    'thread_id': thread_id,
                    'message': message.strip(),
                }
            elif op == 'finish':
                return {
                    'op': op,
                    'thread_id': thread_id,
                    'finished_at': int(data[1]),
                    'node_id': int(data[0]),
                }
            elif op == 'link_back':
                level = int(data[0])
                from_nodes = list(map(int, data[1].split(', '))) if data[1] else []
                to_node = int(data[2])
                return {
                        'op': op,
                        'thread_id': thread_id,
                        'level': level,
                        'from_nodes': from_nodes if from_nodes else [],
                        'to_node': to_node,
                    }
            elif op == 'link_forwards':
                level = int(data[0])
                from_node = int(data[1])
                to_nodes = list(map(int, data[2].split(', '))) if data[2] else []
                return {
                        'op': op,
                        'thread_id': thread_id,
                        'level': level,
                        'from_node': from_node,
                        'to_nodes': to_nodes if to_nodes else [],
                    }
            elif op == 'update_entry':
                return {
                    'op': op,
                    'thread_id': thread_id,
                    'level': int(data[0]),
                    'node_id': int(data[1]),
                }
            elif op == 'unlink':
                return {
                    'op': op,
                    'thread_id': thread_id,
                    'removed_node': int(data[0]),
                    'from_node': int(data[1]),
                }

    # If no pattern matches, return a dictionary with the raw line and thread_id
    return {
        'op': 'unknown',
        'thread_id': thread_id,
        'message': message.strip(),
    }

def generate_ops(file_contents: str) -> List[Dict[str, Any]]:
    ops = []
    for line in file_contents.split('\n'):
        if line:  # skip empty lines
            try:
                result = parse_line(line)
                if isinstance(result, list):
                    ops.extend(result)
                else:
                    ops.append(result)
            except ValueError:
                print(f"Error encountered while parsing line: {line}")
    return ops

def save_to_json(ops: List[Dict[str, Any]], filename: str) -> None:
    with open(filename, 'w') as file:
        json.dump(ops, file, indent=4)


if __name__ == "__main__":
    import sys
    import json

    # Read the log from stdin
    log = sys.stdin.read()

    # Process the log
    ops = generate_ops(log)

    # Write the resulting JSON to stdout
    json.dump(ops, sys.stdout, indent=2)
