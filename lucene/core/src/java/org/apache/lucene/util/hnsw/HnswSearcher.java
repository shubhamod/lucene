/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.util.hnsw;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import java.io.IOException;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.GrowableBitSet;

/**
 * Searches an HNSW graph to find nearest neighbors to a query vector. For more background on the
 * search algorithm, see {@link HnswGraph}.
 *
 * @param <T> the type of query vector
 */
public class HnswSearcher<T> {
  private final VectorSimilarityFunction similarityFunction;
  private final HnswGraph graph;
  private final RandomAccessVectorValues<T> vectors;
  private final VectorEncoding vectorEncoding;
  private SimilarityProvider similarityProvider;

  private final FingerMetadata<T> fingerMetadata;
  /**
   * Scratch data structures that are used in each {@link #searchLevel} call. These can be expensive
   * to allocate, so they're cleared and reused across calls.
   */
  private final NeighborQueue candidates;

  private BitSet visited;

  // provider of distance functions relative to a given query vector
  public interface SimilarityProvider {
    @FunctionalInterface
    interface SimilarityFunction
    {
      float apply(int neighbor) throws IOException;
    }

    // this will perform a full, unoptimized distance calculation against the query
    float exactSimilarityTo(int node) throws IOException;

    /**
     * Returns a function that, given the ordinal of a neighbor of `node`, returns the distance to it.
     * This api allows the FINGER optimizations.
     *
     * @param cNode -- the node about which we will compute similarities
     * @param c2q -- the similarity between the query vector and cNode's vector
     */
    SimilarityFunction approximateSimilarityNear(int cNode, float c2q) throws IOException;
  }

  /**
   * Creates a new graph searcher.
   *
   * @param similarityFunction the similarity function to compare vectors
   * @param visited bit set that will track nodes that have already been visited
   */
  HnswSearcher(
      HnswGraph graph,
      RandomAccessVectorValues<T> vectors,
      VectorEncoding vectorEncoding,
      VectorSimilarityFunction similarityFunction,
      FingerMetadata fingerMetadata,
      BitSet visited) {
    this.graph = graph;
    this.vectors = vectors;
    this.vectorEncoding = vectorEncoding;
    this.similarityFunction = similarityFunction;
    this.fingerMetadata = fingerMetadata;
    this.candidates = new NeighborQueue(100, true);
    this.visited = visited;
  }

  public static class Builder<T> {
    private final HnswGraph graph;
    private final RandomAccessVectorValues<T> vectors;
    private final VectorEncoding vectorEncoding;
    private final VectorSimilarityFunction similarityFunction;
    private FingerMetadata<T> fingerMetadata;
    private boolean concurrent;

    public Builder(HnswGraph graph,
                   RandomAccessVectorValues<T> vectors,
                   VectorEncoding vectorEncoding,
                   VectorSimilarityFunction similarityFunction) {
      this.graph = graph;
      this.vectors = vectors;
      this.vectorEncoding = vectorEncoding;
      this.similarityFunction = similarityFunction;
    }

    public Builder<T> withFinger(FingerMetadata<T> fingerMetadata) {
      this.fingerMetadata = fingerMetadata;
      return this;
    }

    public Builder<T> withConcurrentUpdates() {
      this.concurrent = true;
      return this;
    }

    public HnswSearcher<T> build() {
      BitSet bits = concurrent ? new GrowableBitSet(vectors.size()) : new FixedBitSet(vectors.size());
      return new HnswSearcher<>(graph, vectors, vectorEncoding, similarityFunction, fingerMetadata, bits);
    }
  }

  public NeighborQueue search(
      T query,
      int topK,
      Bits acceptOrds,
      int visitedLimit)
      throws IOException {
    similarityProvider = similarityProviderForQuery(query);

    int initialEp = graph.entryNode();
    if (initialEp == -1) {
      return new NeighborQueue(1, true);
    }
    NeighborQueue results;
    results = new NeighborQueue(1, false);
    int[] eps = new int[] {graph.entryNode()};
    int numVisited = 0;
    for (int level = graph.numLevels() - 1; level >= 1; level--) {
      results.clear();
      searchLevel(results, 1, level, eps, null, visitedLimit);

      numVisited += results.visitedCount();
      visitedLimit -= results.visitedCount();

      if (results.incomplete()) {
        results.setVisitedCount(numVisited);
        return results;
      }
      eps[0] = results.pop();
    }
    results = new NeighborQueue(topK, false);
    searchLevel(
        results, topK, 0, eps, acceptOrds, visitedLimit);
    results.setVisitedCount(results.visitedCount() + numVisited);
    return results;
  }

  private SimilarityProvider similarityProviderForQuery(T query) {
    if (fingerMetadata == null) {
      return new SimilarityProvider() {
        @Override
        public float exactSimilarityTo(int node) throws IOException {
          return compare(query, vectors, node);
        }

        @Override
        public SimilarityFunction approximateSimilarityNear(int cNode, float c2q) throws IOException {
          return neighbor -> compare(query, vectors, neighbor);
        }
      };
    }
    return fingerMetadata.similarityProviderFor(query);
  }

  /**
   * Add the closest neighbors found to a priority queue (heap). These are returned in REVERSE
   * proximity order -- the most distant neighbor of the topK found, i.e. the one with the lowest
   * score/comparison value, will be at the top of the heap, while the closest neighbor will be the
   * last to be popped.
   */
  void searchLevel(
      NeighborQueue results,
      int topK,
      int level,
      final int[] eps,
      Bits acceptOrds,
      int visitedLimit)
      throws IOException {
    assert results.isMinHeap();

    prepareScratchState(vectors.size());

    int numVisited = 0;
    for (int ep : eps) {
      if (visited.getAndSet(ep) == false) {
        if (numVisited >= visitedLimit) {
          results.markIncomplete();
          break;
        }
        float score = similarityProvider.exactSimilarityTo(ep);
        numVisited++;
        candidates.add(ep, score);
        if (acceptOrds == null || acceptOrds.get(ep)) {
          results.add(ep, score);
        }
      }
    }

    // A bound that holds the minimum similarity to the query vector that a candidate vector must
    // have to be considered.
    float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;
    if (results.size() >= topK) {
      minAcceptedSimilarity = results.topScore();
    }
    while (candidates.size() > 0 && results.incomplete() == false) {
      // get the best candidate (closest or best scoring)
      float topCandidateSimilarity = candidates.topScore();
      if (topCandidateSimilarity < minAcceptedSimilarity) {
        break;
      }

      int topCandidateNode = candidates.pop();
      boolean useFinger = numVisited > 5 || level < graph.numLevels() - 2;
      SimilarityProvider.SimilarityFunction localDistances = useFinger
          ? similarityProvider.approximateSimilarityNear(topCandidateNode, topCandidateSimilarity)
          : similarityProvider::exactSimilarityTo;
      graphSeek(graph, level, topCandidateNode);
      int friendOrd;
      while ((friendOrd = graphNextNeighbor(graph)) != NO_MORE_DOCS) {
        if (visited.getAndSet(friendOrd)) {
          continue;
        }

        if (numVisited >= visitedLimit) {
          results.markIncomplete();
          break;
        }
        float friendSimilarity = localDistances.apply(friendOrd);
        numVisited++;
        if (friendSimilarity >= minAcceptedSimilarity) {
          if (useFinger) {
            friendSimilarity = similarityProvider.exactSimilarityTo(friendOrd);
            if (friendSimilarity < minAcceptedSimilarity) {
              continue;
            }
          }
          candidates.add(friendOrd, friendSimilarity);
          if (acceptOrds == null || acceptOrds.get(friendOrd)) {
            if (results.insertWithOverflow(friendOrd, friendSimilarity) && results.size() >= topK) {
              minAcceptedSimilarity = results.topScore();
            }
          }
        }
      }
    }
    while (results.size() > topK) {
      results.pop();
    }
    results.setVisitedCount(numVisited);
  }

  private float compare(T query, RandomAccessVectorValues<T> vectors, int ord) throws IOException {
    if (vectorEncoding == VectorEncoding.BYTE) {
      return similarityFunction.compare((byte[]) query, (byte[]) vectors.vectorValue(ord));
    } else {
      return similarityFunction.compare((float[]) query, (float[]) vectors.vectorValue(ord));
    }
  }

  private void prepareScratchState(int capacity) {
    candidates.clear();
    if (visited.length() < capacity) {
      // this happens during graph construction; otherwise the size of the vector values should
      // be constant, and it will be a SparseFixedBitSet instead of FixedBitSet
      assert (visited instanceof FixedBitSet || visited instanceof GrowableBitSet)
          : "Unexpected visited type: " + visited.getClass().getName();
      if (visited instanceof FixedBitSet) {
        visited = FixedBitSet.ensureCapacity((FixedBitSet) visited, capacity);
      }
      // else GrowableBitSet knows how to grow itself safely
    }
    visited.clear();
  }

  /**
   * Seek a specific node in the given graph. The default implementation will just call {@link
   * HnswGraph#seek(int, int)}
   *
   * @throws IOException when seeking the graph
   */
  void graphSeek(HnswGraph graph, int level, int targetNode) throws IOException {
    graph.seek(level, targetNode);
  }

  /**
   * Get the next neighbor from the graph, you must call {@link #graphSeek(HnswGraph, int, int)}
   * before calling this method. The default implementation will just call {@link
   * HnswGraph#nextNeighbor()}
   *
   * @return see {@link HnswGraph#nextNeighbor()}
   * @throws IOException when advance neighbors
   */
  int graphNextNeighbor(HnswGraph graph) throws IOException {
    return graph.nextNeighbor();
  }
}
