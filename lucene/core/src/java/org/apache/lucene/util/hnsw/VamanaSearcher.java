package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.io.UncheckedIOException;

public class VamanaSearcher <T> {
  private final ThreadLocal<NeighborArray> scratchNeighbors;
  private final ThreadLocal<RandomAccessVectorValues<T>> vectors;
  private final ThreadLocal<RandomAccessVectorValues<T>> vectorsCopy;
  private final ThreadLocal<NeighborQueue> greedyVisitedNodes;
  private final ThreadLocal<FixedBitSet> greedyVisitedSet;
  private final ThreadLocal<FixedNeighborArray> greedyCandidates;

  public VamanaSearcher(ConcurrentVamanaGraph graph, RandomAccessVectorValues<T> ravv, VectorEncoding encoding, VectorSimilarityFunction similarityFunction) {
    this.vectors = ThreadLocal.withInitial(() -> {
      try {
        return ravv.copy();
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    });
    this.vectorsCopy = ThreadLocal.withInitial(() -> {
      try {
        return ravv.copy();
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    });
    scratchNeighbors = ThreadLocal.withInitial(() -> new NeighborArray(16, true));
    greedyVisitedNodes = ThreadLocal.withInitial(() -> new NeighborQueue(16, false));
    greedyVisitedSet = ThreadLocal.withInitial(() -> new FixedBitSet(graph.size()));
    this.greedyCandidates = ThreadLocal.withInitial(() -> new FixedNeighborArray(16));
  }


}
