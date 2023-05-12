import sys
import re

# Define the start and end strings
start_construction_str = r"selective ords"
end_construction_str = r"Found results"
end_graph_str = "java.lang.AssertionError: "

# Initialize the flags
in_construction = False
in_graph = False
skip_next_line = False

# Open the input and output files
with open("badgraph-construction.txt", "w") as construction_file, \
        open("badgraph.txt", "w") as graph_file:
    # Read the log file line by line
    for line in sys.stdin:
        if skip_next_line:
            skip_next_line = False
            continue
        # Check if the line matches the start or end strings
        if start_construction_str in line:
            in_construction = True
            skip_next_line = True
            continue
        elif end_construction_str in line:
            in_construction = False
            in_graph = True
        elif line.startswith(end_graph_str):
            in_graph = False
            break

        # Write the line to the corresponding output file
        if in_construction:
            construction_file.write(line)
            skip_start_line = False
        elif in_graph:
            graph_file.write(line)
