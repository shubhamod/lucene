package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.GrowableBitSet;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.LongAdder;
import java.util.stream.IntStream;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * VamanaGraphBuilder builds a VamanaGraph from an HNSW graph.
 */
public class VamanaGraphBuilder<T> {
  private final VectorEncoding encoding;
  private final VectorSimilarityFunction similarityFunction;
  private final int beamWidth;
  private final HnswGraph hnsw;

  private final ThreadLocal<NeighborArray> scratchNeighbors;
  private final ThreadLocal<RandomAccessVectorValues<T>> vectors;
  private final ThreadLocal<RandomAccessVectorValues<T>> vectorsCopy;
  private final ThreadLocal<NeighborQueue> greedyVisitedNodes;
  private final ThreadLocal<FixedBitSet> greedyVisitedSet;
  private final ThreadLocal<FixedNeighborArray> greedyCandidates;

  public VamanaGraphBuilder(HnswGraph hnsw, RandomAccessVectorValues<T> ravv, VectorEncoding encoding, VectorSimilarityFunction similarityFunction, int beamWidth) {
    this.hnsw = hnsw;
    this.encoding = encoding;
    this.similarityFunction = similarityFunction;
    this.beamWidth = beamWidth;
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
    scratchNeighbors = ThreadLocal.withInitial(() -> new NeighborArray(beamWidth, true));
    greedyVisitedNodes = ThreadLocal.withInitial(() -> new NeighborQueue(beamWidth, false));
    greedyVisitedSet = ThreadLocal.withInitial(() -> new FixedBitSet(hnsw.size()));
    this.greedyCandidates = ThreadLocal.withInitial(() -> new FixedNeighborArray(beamWidth));
  }


  /**
   * For testing.  Bad vanama is just L0 of the hnsw graph.
   */
  ConcurrentVamanaGraph buildBadVamana(float alpha) throws IOException {
    ConcurrentOnHeapHnswGraph chnsw = (ConcurrentOnHeapHnswGraph) hnsw;
    int s = hnsw.entryNode();
    return new ConcurrentVamanaGraph(hnsw, chnsw.nsize0, s, chnsw.similarity);
  }

  /**
   * For testing.  Quadratic!
   */
  ConcurrentVamanaGraph buildOptimal(float alpha) throws IOException {
    int s = hnsw.entryNode();
    ConcurrentNeighborSet.NeighborSimilarity sf = similarityFunction();
    var optimal = new ConcurrentVamanaGraph(hnsw, hnsw.size() - 1, s, sf);
    IntStream.range(0, hnsw.size()).parallel().forEach(i -> {
      // potential neighbors = everyone
      NeighborArray world = new NeighborArray(hnsw.size() - 1, true);
      var similarity = sf.scoreProvider(i);
      for (int j = 0; j < hnsw.size(); j++) {
        if (i == j) {
          continue;
        }
        world.insertSorted(j, similarity.apply(j));
      }

      optimal.getNeighbors(i).robustPrune(world, alpha);
    });
    return optimal;
  }


    static String prettyPrint(HnswGraph hnsw) {
    StringBuilder sb = new StringBuilder();
    sb.append(hnsw);
    sb.append("\n");

    try {
      for (int level = 0; level < hnsw.numLevels(); level++) {
        sb.append("# Level ").append(level).append("\n");
        HnswGraph.NodesIterator it = hnsw.getNodesOnLevel(level);
        while (it.hasNext()) {
          int node = it.nextInt();
          sb.append("  ").append(node).append(" -> ");
          hnsw.seek(level, node);
          while (true) {
            int neighbor = hnsw.nextNeighbor();
            if (neighbor == NO_MORE_DOCS) {
              break;
            }
            sb.append(" ").append(neighbor);
          }
          sb.append("\n");
        }
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    return sb.toString();
  }

  /**
   * buildVamana assumes that the graph is completely built and no further nodes
   * are added concurrently.
   */
  public ConcurrentVamanaGraph buildVamana(int M, float alpha) {
    int s = approximateMediod();
    System.out.println("mediod is " + s);
    ConcurrentVamanaGraph vamana = new ConcurrentVamanaGraph(hnsw, M, s, similarityFunction());

    // iterate over the points in a random order.
    List<Integer> L = generateRandomPermutation(vamana.size());
    LongAdder nodesVisited = new LongAdder();
    LongAdder edgesChanged = new LongAdder();
    L.stream().parallel().filter(p -> p != s).forEach(p -> {
      try {
        NeighborQueue cq = greedySearch(vamana, p);
        nodesVisited.add(cq.size());
        NeighborArray candidates = popToDesc(cq);
        NeighborArray added = vamana.getNeighbors(p).robustPrune(candidates, alpha);
        edgesChanged.add(added.size);
        for (int i = 0; i < added.size(); i++) {
          int q = added.node[i];
          assert q != p : "node " + p + " should not be a neighbor of itself";
          vamana.getNeighbors(q).insert(p, added.score[i], alpha);
        }
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    });

    System.out.printf("visited %d nodes and changed %d edges to build vamana index for graph of %d%n",
        nodesVisited.sum(), edgesChanged.sum(), hnsw.size());
    return vamana;
  }

  private ConcurrentNeighborSet.NeighborSimilarity similarityFunction() {
    return new ConcurrentNeighborSet.NeighborSimilarity() {
      @Override
      public float score(int node1, int node2) {
        try {
          return scoreBetween(
              vectors.get().vectorValue(node1), vectorsCopy.get().vectorValue(node2));
        } catch (IOException e) {
          throw new UncheckedIOException(e);
        }
      }

      @Override
      public ScoreFunction scoreProvider(int node1) {
        T v1;
        try {
          v1 = vectors.get().vectorValue(node1);
        } catch (IOException e) {
          throw new UncheckedIOException(e);
        }
        return node2 -> {
          try {
            return scoreBetween(v1, vectorsCopy.get().vectorValue(node2));
          } catch (IOException e) {
            throw new UncheckedIOException(e);
          }
        };
      }
    };
  }

  /**
   * Returns *all* the nodes visited by a greedy best-first search from s to p.
   * (NOT the top k nodes!)
   */
  private NeighborQueue greedySearch(ConcurrentVamanaGraph vamana, int p) throws IOException {
    var resultCandidates = greedyCandidates.get();
    resultCandidates.clear();
    var visitedSet = greedyVisitedSet.get();
    visitedSet.clear();
    var visitedNodes = greedyVisitedNodes.get();
    visitedNodes.clear();
    var v = vectors.get();
    var vc = vectorsCopy.get();
    var graph = vamana.getView();

    int s = vamana.entryNode();
    T vP = v.vectorValue(p);
    resultCandidates.push(s, scoreBetween(vP, vc.vectorValue(s)));
    visitedSet.set(p);
    while (true) {
      // get the best candidate (closest or best scoring)
      int n = resultCandidates.nextUnvisited(visitedSet);
      if (n < 0) {
        break;
      }

      int topCandidateNode = resultCandidates.node[n];
      float topCandidateSimilarity = resultCandidates.score[n];
      visitedNodes.add(topCandidateNode, topCandidateSimilarity);
      visitedSet.set(topCandidateNode);
      graph.seek(0, topCandidateNode);
      int friendOrd;
      while ((friendOrd = graph.nextNeighbor()) != NO_MORE_DOCS) {
        if (visitedSet.get(friendOrd)) {
          continue;
        }
        assert friendOrd != p : "node " + p + " should not be a neighbor of itself";

        T vNeighbor = vc.vectorValue(friendOrd);
        float friendSimilarity = scoreBetween(vP, vNeighbor);
        resultCandidates.push(friendOrd, friendSimilarity);
      }
    }
    return visitedNodes;
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

  // TODO make a searcher object, make the threadlocals instance fields there
  public QueryResult greedySearch(ConcurrentVamanaGraph vamana, T vP, int topK) throws IOException {
    var resultCandidates = new FixedNeighborArray(topK);
    var visitedSet = greedyVisitedSet.get();
    visitedSet.clear();
    var visitedNodesCount = 0;
    var vc = vectorsCopy.get();
    var graph = vamana.getView();

    int s = vamana.entryNode();
    resultCandidates.push(s, scoreBetween(vP, vc.vectorValue(s)));
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

        T vNeighbor = vc.vectorValue(friendOrd);
        float friendSimilarity = scoreBetween(vP, vNeighbor);
        resultCandidates.push(friendOrd, friendSimilarity);
      }
    }
    return new QueryResult(resultCandidates, visitedNodesCount);
  }

  private List<Integer> generateRandomPermutation(int size) {
    var L = new ArrayList<Integer>(size);
    for (int i = 0; i < size; i++) {
      L.add(i);
    }
    Collections.shuffle(L);
    return L;
  }

  // the mediod doesn't seem to matter a whole lot in practice -- the results are nearly the same if we use
  // the hnsw entry node which is effectively random.  this is fortunate because I don't know how to do
  // an approximation that is both high-quality and fast.
  private int approximateMediod() {
    assert hnsw.size() > 0;
    // pick the best mediod candidate from beamWidth options, each of which uses its nearest neighbors to approximate
    // the true distance to all neighbors
    IntStream range = IntStream.range(0, beamWidth);
    if (hnsw instanceof ConcurrentOnHeapHnswGraph) {
      range = range.parallel();
    }
    return range
        .mapToObj(
            n -> {
              int i = ThreadLocalRandom.current().nextInt(hnsw.size());
              HnswGraphSearcher<T> searcher = new HnswGraphSearcher<>(
                  encoding,
                  similarityFunction,
                  new NeighborQueue(beamWidth, true),
                  new GrowableBitSet(this.vectors.get().size()));
              NeighborQueue neighbors;
              try {
                var graph = hnsw instanceof ConcurrentOnHeapHnswGraph ? ((ConcurrentOnHeapHnswGraph) hnsw).getView() : hnsw;
                neighbors = HnswGraphSearcher.search(vectors.get().vectorValue(i), beamWidth, vectorsCopy.get(), graph, searcher, null, Integer.MAX_VALUE);
              } catch (IOException e) {
                throw new RuntimeException(e);
              }
              // compute average distance to the neighbors
              float sum = 0;
              for (int j = 0; j < neighbors.size(); j++) {
                sum += neighbors.topScore();
                neighbors.pop();
              }
              return new AbstractMap.SimpleEntry<>(i, sum / neighbors.size());
            }).min(java.util.Map.Entry.comparingByValue())
        // index of the node with the smallest average distance
        .get().getKey();
  }

  protected float scoreBetween(T v1, T v2) {
    return switch (encoding) {
      case BYTE -> similarityFunction.compare((byte[]) v1, (byte[]) v2);
      case FLOAT32 -> similarityFunction.compare((float[]) v1, (float[]) v2);
    };
  }

  private NeighborArray popToDesc(NeighborQueue candidates) {
    NeighborArray scratch = this.scratchNeighbors.get();
    scratch.clear();
    while (scratch.node.length < candidates.size()) {
      scratch.growArrays();
    }

    // the neighbors will be popped from worst to best, so we reverse that
    // by reaching into the private fields of the NeighborQueue
    int candidateCount = candidates.size();
    for (int i = 0; i < candidateCount; i++) {
      float similarity = candidates.topScore();
      int node = candidates.pop();
      int n = candidateCount - i - 1;
      scratch.node[n] = node;
      scratch.score[n] = similarity;
    }
    scratch.size = candidateCount;
    return scratch;
  }
}
