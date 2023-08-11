package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;

import java.io.IOException;
import java.io.UncheckedIOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class VamanaSearcher <T> {
  private final VectorEncoding encoding;
  private final VectorSimilarityFunction similarityFunction;
  private final RandomAccessVectorValues<T> vectors;
  private final RandomAccessVectorValues<T> vectorsCopy;

  private final NeighborArray scratchNeighbors;
  private final NeighborQueue greedyVisitedNodes;
  private final BitSet visitedSet;
  private final FixedNeighborArray greedyCandidates;

  public VamanaSearcher(ConcurrentVamanaGraph graph, RandomAccessVectorValues<T> ravv, VectorEncoding encoding, VectorSimilarityFunction similarityFunction) {
    this.encoding = encoding;
    this.similarityFunction = similarityFunction;
    try {
      this.vectors = ravv.copy();
      this.vectorsCopy = ravv.copy();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    scratchNeighbors = new NeighborArray(16, true);
    greedyVisitedNodes = new NeighborQueue(16, false);
    visitedSet = new FixedBitSet(graph.size());
    this.greedyCandidates = new FixedNeighborArray(16);
  }

  protected float scoreBetween(T v1, T v2) {
    return switch (encoding) {
      case BYTE -> similarityFunction.compare((byte[]) v1, (byte[]) v2);
      case FLOAT32 -> similarityFunction.compare((float[]) v1, (float[]) v2);
    };
  }

  /** Query result. */
  public static final class QueryResult {
    public final FixedNeighborArray results;
    public final long visitedCount;

    public QueryResult(FixedNeighborArray results, long visitedCount) {
      this.results = results;
      this.visitedCount = visitedCount;
    }
  }

  public VamanaSearcher.QueryResult search(ConcurrentVamanaGraph vamana, T vP, int topK) throws IOException {
    var resultCandidates = new FixedNeighborArray(topK);
    visitedSet.clear();
    var visitedNodesCount = 0;
    var graph = vamana.getView();

    int s = vamana.entryNode();
    resultCandidates.push(s, scoreBetween(vP, vectorsCopy.vectorValue(s)));
    while (true) {
      // get the best candidate (closest or best scoring)
      int n = resultCandidates.nextUnvisited(visitedSet);
      if (n < 0) {
        break;
      }

      int topCandidateNode = resultCandidates.node[n];
      visitedNodesCount++;
      visitedSet.set(topCandidateNode);
      graph.seek(0, topCandidateNode);
      int friendOrd;
      while ((friendOrd = graph.nextNeighbor()) != NO_MORE_DOCS) {
        if (visitedSet.get(friendOrd)) {
          continue;
        }

        T vNeighbor = vectorsCopy.vectorValue(friendOrd);
        float friendSimilarity = scoreBetween(vP, vNeighbor);
        resultCandidates.push(friendOrd, friendSimilarity);
      }
    }
    return new VamanaSearcher.QueryResult(resultCandidates, visitedNodesCount);
  }
}
