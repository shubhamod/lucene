import sys

# Define the start and end strings
start_construction_str = "--- selective ords docs=60 accept=5 ---"
end_construction_str = "Node 0 cannot reach [48] on level 0 in ConcurrentOnHeapHnswGraphView(size=80, entryPoint=NodeAtLevel(level=5, node=10)"
start_graph_str = end_construction_str
end_graph_str = "java.lang.AssertionError: " + end_construction_str

# Initialize the flags
in_construction = False
in_graph = False
skip_start_line = True

# Open the input and output files
with open("badgraph-construction.txt", "w") as construction_file, \
        open("badgraph.txt", "w") as graph_file:
    # Read the log file line by line
    for line in sys.stdin:
        # Check if the line matches the start or end strings
        if line.strip() == start_construction_str:
            in_construction = True
            skip_start_line = True
            continue
        elif line.strip() == end_construction_str:
            in_construction = False
            in_graph = True
        elif line.startswith(end_graph_str):
            in_graph = False

        # Write the line to the corresponding output file
        if in_construction and not skip_start_line:
            construction_file.write(line)
        elif in_graph:
            graph_file.write(line)

        # Reset the skip_start_line flag after writing the first line in construction
        if in_construction and skip_start_line:
            skip_start_line = False
