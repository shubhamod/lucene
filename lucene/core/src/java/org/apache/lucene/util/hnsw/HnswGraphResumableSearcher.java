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

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.SparseFixedBitSet;

import java.io.IOException;
import java.util.ArrayList;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * Searches HNSW graph for the nearest neighbors of a query vector.  This API is intended
 * to be used in a "microbatching" streaming query pipeline that includes non-HNSW predicates.  The
 * goal is to be able to "resume" a search if the first call does not find enough results,
 * without having to re-scan the same nodes again.
 * <p>
 * If you can always get the results you need in a single search call, it will be more
 * efficient to use HnswGraphSearcher instead, because this class can't optimize as aggressively. In
 * particular, it needs to keep track of candidates that are worse than the best topK
 * found so far, because it might need to use those in a subsequent call.
 */
public class HnswGraphResumableSearcher<T> {
  private final VectorSimilarityFunction similarityFunction;
  private final VectorEncoding vectorEncoding;

  private final HnswGraph graph;
  private final Bits acceptOrds;
  private final NeighborQueue searchSpace;
  private NeighborQueue results;

  // initialized in search because it's easier to make Java's generics happy there
  private final T query;
  private final RandomAccessVectorValues<T> vectors;

  private BitSet visited;

  /**
   * Creates a new graph searcher.
   *
   * @param searchSpace the priority queue of nodes remaining to search
   * @param similarityFunction the similarity function to compare vectors
   * @param graph the graph values. May represent the entire graph, or a level in a hierarchical
   *     graph.
   * @param acceptOrds {@link Bits} that represents the allowed document ordinals to match, or
   *     {@code null} if they are all allowed to match.
   * @param visited bitset to track visited nodes across calls
   */
  public HnswGraphResumableSearcher(
          T query,
            RandomAccessVectorValues<T> vectors,
        NeighborQueue searchSpace,
          VectorEncoding vectorEncoding,
          VectorSimilarityFunction similarityFunction,
          HnswGraph graph,
          Bits acceptOrds,
          BitSet visited) {
    if (query instanceof float[]) {
      if (((float[]) query).length != vectors.dimension()) {
        throw new IllegalArgumentException("query vector dimension " + ((float[]) query).length + " != " + vectors.dimension());
      }
    } else if (query instanceof byte[]) {
      if (((byte[]) query).length != vectors.dimension()) {
        throw new IllegalArgumentException("query vector dimension " + ((byte[]) query).length + " != " + vectors.dimension());
      }
    } else {
        throw new IllegalArgumentException("vectors must be float[] or byte[]");
    }

    this.query = query;
    this.vectors = vectors;
    this.searchSpace = searchSpace;
    this.vectorEncoding = vectorEncoding;
    this.similarityFunction = similarityFunction;
    this.graph = graph;
    this.acceptOrds = acceptOrds;
    this.visited = visited;
  }

  /**
   * Perform the initial search for a query vector.
   * <p>
   * Stops searching when the next-best node is worse than the `topK` found so far,
   * or when it hits `visitLimit`.
   *
   * Neighbors are returned with the WORST of the topK neighbors at the top of the queue;
   * pop() them off to get them sorted worst-to-best.
   */
  public NeighborQueue search(int topK, int visitLimit) throws IOException {
    // first, follow the index until we get to level 0
    HnswGraphSearcher<T> levelSearcher = new HnswGraphSearcher<>(vectorEncoding,
            similarityFunction,
            new NeighborQueue(1, true),
            new SparseFixedBitSet(graph.size()));
    int initialEp = graph.entryNode();
    if (initialEp == -1) {
      return new NeighborQueue(1, true);
    }
    int[] eps = new int[] {graph.entryNode()};
    for (int level = graph.numLevels() - 1; level >= 1; level--) {
      levelSearcher.searchLevel(query, 1, level, eps, vectors, graph, null, visitLimit);
      eps[0] = results.pop();
    }

    // level 0 search
    return resume(topK, visitLimit);
  }

  /**
   * Resume a search after the initial call to `search`.
   * <p>
   * Stops searching when the next-best node is worse than the `topK` found so far,
   * or when it hits `visitLimit`.
   */
  public NeighborQueue resume(int topK, int visitLimit) throws IOException {
    // In pseudocode, resume looks like this:
    //
    // # visited: a set of already-visited nodes in the search space
    // # candidates: priority queue of nodes to consider for topK
    // # searchSpace: priority queue of nodes that we know about in the area of our target
    // # visitLimit: consider at least this many nodes
    // resume(topK, visited, candidates, searchSpace, visitLimit):
    //   nVisited = 0
    //   while nVisited < visitLimit:
    //    N = searchSpace.pop()
    //    if N not in visited:
    //      nVisited++
    //      visited.add(N)
    //      candidates.push(N)
    //      searchSpace.pushAll(N.neighbors())
    //    [caller can retrieve top candidates from the queue]
    if (results == null || results.capacity() != topK) {
        results = new NeighborQueue(topK, true);
    } else {
      results.clear();
    }

    float minAcceptedSimilarity = Float.NEGATIVE_INFINITY;
    ArrayList<Integer> discarded = new ArrayList<>();
    int numVisited = 0;
    while (numVisited++ < visitLimit && searchSpace.size() > 0) {
      int friendOrd = searchSpace.pop();
      visited.set(friendOrd);
      float friendSimilarity = compare(query, vectors, friendOrd);
      if (acceptOrds == null || acceptOrds.get(friendOrd)) {
        var oldTop = results.topNode();
        if (friendSimilarity > minAcceptedSimilarity) {
          var r = results.insertWithOverflow(friendOrd, friendSimilarity);
          assert r : "thought we were higher priority than the worst on the heap, but were not";
          discarded.add(oldTop);
          if (results.size() == topK) {
            minAcceptedSimilarity = results.topScore();
          }
        } else {
            discarded.add(friendOrd);
        }
      }

      graph.seek(0, friendOrd);
      while ((friendOrd = graph.nextNeighbor()) != NO_MORE_DOCS) {
        if (!visited.get(friendOrd)) {
          searchSpace.add(friendOrd, compare(query, vectors, friendOrd));
        }
      }
    }

    // add nodes that weren't good enough back to the search space for next time
    for (int discardedOrd : discarded) {
      searchSpace.add(discardedOrd, compare(query, vectors, discardedOrd));
    }

    return results;
  }

  public static <T> NeighborQueue search(
          T query,
          int topK,
          RandomAccessVectorValues<T> vectors,
          VectorEncoding vectorEncoding,
          VectorSimilarityFunction similarityFunction,
          HnswGraph graph,
          Bits acceptOrds,
          int visitedLimit)
          throws IOException {
    NeighborQueue searchSpace = new NeighborQueue(topK, true);
    BitSet visited = new SparseFixedBitSet(graph.size());

    HnswGraphResumableSearcher<T> resumableSearcher = new HnswGraphResumableSearcher<>(
            query,
            vectors,
            searchSpace,
            vectorEncoding,
            similarityFunction,
            graph,
            acceptOrds,
            visited);
    return resumableSearcher.search(topK, visitedLimit);
  }

  private float compare(T query, RandomAccessVectorValues<T> vectors, int ord) throws IOException {
    if (vectorEncoding == VectorEncoding.BYTE) {
      return similarityFunction.compare((byte[]) query, (byte[]) vectors.vectorValue(ord));
    } else {
      return similarityFunction.compare((float[]) query, (float[]) vectors.vectorValue(ord));
    }
  }
}
