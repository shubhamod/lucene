import sys
import re

# Define the start and end strings
start_construction_str = r"\d+ adding node \d+"
end_construction_str = r"ConcurrentOnHeapHnswGraphView"
start_graph_str = "# Level 0"
end_graph_str = "Compare serial"

# Initialize the flags
in_construction = False
in_graph = False

# Open the input and output files
with open("badgraph-construction.txt", "w") as construction_file, \
        open("badgraph.txt", "w") as graph_file:
    # Read the log file line by line
    for line in sys.stdin:
        # Check if the line matches the start or end strings
        if not in_construction and re.match(start_construction_str, line):
            in_construction = True
        elif in_construction and end_construction_str in line:
            in_construction = False
            continue
        elif line.startswith(start_graph_str):
            in_graph = True
            continue
        elif line.startswith(end_graph_str):
            in_graph = False
            break

        # Write the line to the corresponding output file
        if in_construction:
            construction_file.write(line)
        elif in_graph:
            graph_file.write(line)
