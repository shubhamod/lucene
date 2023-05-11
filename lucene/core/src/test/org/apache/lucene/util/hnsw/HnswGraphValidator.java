package org.apache.lucene.util.hnsw;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class HnswGraphValidator {
  private final HnswGraph hnsw;

  public HnswGraphValidator(HnswGraph hnsw) {
    this.hnsw = hnsw instanceof ConcurrentOnHeapHnswGraph ? ((ConcurrentOnHeapHnswGraph) hnsw).getView() : hnsw;
  }

  public void validateReachability() throws IOException {
    for (int level = 0; level < hnsw.numLevels(); level++) {
      validateReachability(level);
    }
  }

  private void validateReachability(int level) throws IOException {
    Set<Integer> nodes = getAllNodes(level);
    for (Integer node : nodes) {
      validateNodeCanReachOthers(node, level, new HashSet<>(nodes));
    }
  }

  private Set<Integer> getAllNodes(int level) throws IOException {
    HnswGraph.NodesIterator nodesIterator = hnsw.getNodesOnLevel(level);
    Set<Integer> nodes = new HashSet<>();
    while (nodesIterator.hasNext()) {
      nodes.add(nodesIterator.nextInt());
    }
    return nodes;
  }

  private void validateNodeCanReachOthers(Integer startNode, int level, Set<Integer> remaining) throws IOException {
    dfs(startNode, level, remaining);
    assert remaining.isEmpty() : "Node " + startNode + " cannot reach " + remaining + " in " + ConcurrentHnswGraphTestCase.prettyPrint(hnsw);
  }

  private void dfs(Integer node, int level, Set<Integer> remaining) throws IOException {
    remaining.remove(node);
    hnsw.seek(level, node);
    for (int neighbor = hnsw.nextNeighbor(); neighbor != NO_MORE_DOCS; neighbor = hnsw.nextNeighbor()) {
      if (remaining.contains(neighbor)) {
        dfs(neighbor, level, remaining);
      }
    }
  }
}

