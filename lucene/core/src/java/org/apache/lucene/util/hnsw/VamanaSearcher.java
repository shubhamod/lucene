package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.GrowableBitSet;

import java.io.IOException;
import java.io.UncheckedIOException;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/** beamwidth-based searcher */
public class VamanaSearcher <T> {
  private final ConcurrentOnHeapHnswGraph graph;
  private final VectorEncoding encoding;
  private final VectorSimilarityFunction similarityFunction;
  private final RandomAccessVectorValues<T> vectors;

  private final BitSet visitedSet;

  public VamanaSearcher(ConcurrentOnHeapHnswGraph graph, RandomAccessVectorValues<T> ravv, VectorEncoding encoding, VectorSimilarityFunction similarityFunction) {
    this.graph = graph;
    this.encoding = encoding;
    this.similarityFunction = similarityFunction;
    try {
      this.vectors = ravv.copy();
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
    visitedSet = new GrowableBitSet(graph.size());
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

  public QueryResult search(T vP, int beamWidth) throws IOException {
    return search(vP, beamWidth, null, null);
  }

  public QueryResult search(T vP, int beamWidth, Bits acceptOrds, NeighborQueue visitedNodes) throws IOException {
    var resultCandidates = new FixedNeighborArray(beamWidth);
    visitedSet.clear();
    var visitedNodesCount = 0;
    var view = graph.getView();

    int s = graph.entryNode();
    if (s < 0) {
      return new QueryResult(resultCandidates, visitedNodesCount);
    }

    resultCandidates.push(s, scoreBetween(vP, vectors.vectorValue(s)));
    while (true) {
      // get the best candidate (closest or best scoring)
      int n = resultCandidates.nextUnvisited(visitedSet);
      if (n < 0) {
        break;
      }
      int topCandidateNode = resultCandidates.node[n];

      // stats
      visitedNodesCount++;
      if (visitedNodes != null) {
        visitedNodes.add(topCandidateNode, resultCandidates.score[n]);
      }
      visitedSet.set(topCandidateNode);

      // add neighbors to resultCandidates, if they are good enough
      view.seek(0, topCandidateNode);
      int friendOrd;
      while ((friendOrd = view.nextNeighbor()) != NO_MORE_DOCS) {
        if (visitedSet.get(friendOrd) || (acceptOrds != null && !acceptOrds.get(friendOrd))) {
          continue;
        }

        T vNeighbor = vectors.vectorValue(friendOrd);
        float friendSimilarity = scoreBetween(vP, vNeighbor);
        resultCandidates.push(friendOrd, friendSimilarity);
      }
    }

    return new QueryResult(resultCandidates, visitedNodesCount);
  }
}
