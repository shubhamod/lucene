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

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.PrimitiveIterator;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.FixedBitSet;

/** A concurrent set of neighbors. */
public class ConcurrentNeighborSet {
  /** the node id whose neighbors we are storing */
  private final int nodeId;

  /**
   * We use a copy-on-write NeighborArray to store the neighbors. Even though updating this is
   * expensive, it is still faster than using a concurrent Collection because "iterate through a
   * node's neighbors" is a hot loop in adding to the graph, and NeighborArray can do that much
   * faster: no boxing/unboxing, all the data is stored sequentially instead of having to follow
   * references, and no fancy encoding necessary for node/score.
   */
  private final AtomicReference<ConcurrentNeighborArray> neighborsRef;

  private final NeighborSimilarity similarity;

  /** the maximum number of neighbors we can store */
  private final int maxConnections;

  public ConcurrentNeighborSet(int nodeId, int maxConnections, NeighborSimilarity similarity) {
    this.nodeId = nodeId;
    this.maxConnections = maxConnections;
    this.similarity = similarity;
    neighborsRef = new AtomicReference<>(new ConcurrentNeighborArray(maxConnections, true));
  }

  private ConcurrentNeighborSet(ConcurrentNeighborSet old) {
    this.nodeId = old.nodeId;
    this.maxConnections = old.maxConnections;
    this.similarity = old.similarity;
    neighborsRef = new AtomicReference<>(old.neighborsRef.get());
  }

  public ConcurrentNeighborSet(int i, NeighborArray neighbors, int maxConnections, NeighborSimilarity similarity) {
    this.nodeId = i;
    this.maxConnections = maxConnections;
    this.similarity = similarity;
    ConcurrentNeighborArray cna = new ConcurrentNeighborArray(maxConnections, true);
    for (int j = 0; j < neighbors.size(); j++) {
      cna.addInOrder(neighbors.node[j], neighbors.score[j]);
    }
    neighborsRef = new AtomicReference<>(cna);
  }

  public PrimitiveIterator.OfInt nodeIterator() {
    // don't use a stream here. stream's implementation of iterator buffers
    // very aggressively, which is a big waste for a lot of searches.
    return new NeighborIterator(neighborsRef.get());
  }

  public void backlink(Function<Integer, ConcurrentNeighborSet> neighborhoodOf) throws IOException {
    NeighborArray neighbors = neighborsRef.get();
    for (int i = 0; i < neighbors.size(); i++) {
      int nbr = neighbors.node[i];
      float nbrScore = neighbors.score[i];
      ConcurrentNeighborSet nbrNbr = neighborhoodOf.apply(nbr);
      nbrNbr.insert(nodeId, nbrScore);
    }
  }

  private static class NeighborIterator implements PrimitiveIterator.OfInt {
    private final NeighborArray neighbors;
    private int i;

    private NeighborIterator(NeighborArray neighbors) {
      this.neighbors = neighbors;
      i = 0;
    }

    @Override
    public boolean hasNext() {
      return i < neighbors.size();
    }

    @Override
    public int nextInt() {
      return neighbors.node[i++];
    }
  }

  public int size() {
    return neighborsRef.get().size();
  }

  public int arrayLength() {
    return neighborsRef.get().node.length;
  }

  /**
   * For each candidate (going from best to worst), select it only if it is closer to target than it
   * is to any of the already-selected candidates. This is maintained whether those other neighbors
   * were selected by this method, or were added as a "backlink" to a node inserted concurrently
   * that chose this one as a neighbor.
   */
  public void insertDiverse(NeighborArray candidates) {
    BitSet selected = new FixedBitSet(candidates.size());
    for (int i = candidates.size() - 1; i >= 0; i--) {
      int cNode = candidates.node[i];
      float cScore = candidates.score[i];
      if (isDiverse(cNode, cScore, candidates, selected, 1.0f)) {
        selected.set(i);
      }
    }
    insertMultiple(candidates, selected);
    // This leaves the paper's keepPrunedConnection option out; we might want to add that
    // as an option in the future.
  }

  public ConcurrentNeighborArray getCurrent() {
    return neighborsRef.get();
  }

  /**
   * Set the neighbors to the result of the Vamana RobustPrune algorithm.  In a single-threaded
   * context, we would expect candidates to be a superset of the existing neighbors, but
   * since we are running this concurrently, another thread may have modified the existing neighbors
   * and we want to make sure not to throw those away unnecessarily.
   *
   * @return an array of the neighbors that were actually added
   */
  public NeighborArray robustPrune(INeighborArray externalCandidates, float alpha) {
    // this is basically insertDiverse + insertMultiple, only wrapped into a coarser getAndUpdate
    NeighborArray old = neighborsRef.get();
    NeighborArray next = neighborsRef.updateAndGet(current -> {
      NeighborArray candidates = mergeCandidates(externalCandidates, current);
      assert !candidates.scoresDescOrder;
      BitSet selected = new FixedBitSet(candidates.size());
      for (int i = candidates.size() - 1; i >= 0 && selected.cardinality() < maxConnections; i--) {
        int cNode = candidates.node[i];
        float cScore = candidates.score[i];
        if (isDiverse(cNode, cScore, candidates, selected, alpha)) {
          selected.set(i);
        }
      }

      ConcurrentNeighborArray na = new ConcurrentNeighborArray(maxConnections, true);
      for (int i = candidates.size() - 1; i >= 0; i--) {
        if (!selected.get(i)) {
          continue;
        }
        int node = candidates.node[i];
        float score = candidates.score[i];
        na.insertSorted(node, score);
      }
      return na;
    });

    // TODO can we do better than this?
    NeighborArray added = new NeighborArray(maxConnections, true);
    for (int i = 0; i < next.size(); i++) {
      if (!containsNode(old, next.node[i])) {
        added.addInOrder(next.node[i], next.score[i]);
      }
    }
    return added;
  }

  private static boolean containsNode(NeighborArray na, int node) {
    for (int i = 0; i < na.size(); i++) {
      if (na.node[i] == node) {
        return true;
      }
    }
    return false;
  }

  static NeighborArray mergeCandidates(INeighborArray a1, NeighborArray a2) {
    assert a1.scoresDescending();
    assert a2.scoresDescending();

    NeighborArray merged = new NeighborArray(a1.size() + a2.size(), true);
    int i = 0, j = 0;

    while (i < a1.size() && j < a2.size()) {
      if (a1.score()[i] < a2.score[j]) {
        merged.addInOrder(a2.node[j], a2.score[j]);
        j++;
      } else if (a1.score()[i] > a2.score[j]) {
        merged.addInOrder(a1.node()[i], a1.score()[i]);
        i++;
      } else {
        merged.addInOrder(a1.node()[i], a1.score()[i]);
        if (a2.node[j] != a1.node()[i]) {
          merged.addInOrder(a2.node[j], a2.score[j]);
        }
        i++;
        j++;
      }
    }

    // If elements remain in a1, add them
    while (i < a1.size()) {
      // Skip duplicates between the remaining elements in a1 and the last added element in a2
      if (j > 0 && i < a1.size() && a1.node()[i] == a2.node[j-1]) {
        i++;
        continue;
      }
      merged.addInOrder(a1.node()[i], a1.score()[i]);
      i++;
    }

    // If elements remain in a2, add them
    while (j < a2.size()) {
      // Skip duplicates between the remaining elements in a2 and the last added element in a1
      if (i > 0 && j < a2.size() && a2.node[j] == a1.node()[i-1]) {
        j++;
        continue;
      }
      merged.addInOrder(a2.node[j], a2.score[j]);
      j++;
    }

    // TODO fixme
    var m2 = new NeighborArray(merged.size(), false);
    for (int k = merged.size() - 1; k >= 0; k--) {
      m2.addInOrder(merged.node[k], merged.score[k]);
    }
    return m2;
  }

  private void insertMultiple(NeighborArray others, BitSet selected) {
    neighborsRef.getAndUpdate(
        current -> {
          ConcurrentNeighborArray next = current.copy();
          for (int i = others.size() - 1; i >= 0; i--) {
            if (!selected.get(i)) {
              continue;
            }
            int node = others.node[i];
            float score = others.score[i];
            next.insertSorted(node, score);
          }
          enforceMaxConnLimit(next, 1.0f);
          return next;
        });
  }

  /**
   * Insert a new neighbor, maintaining our size cap by removing the least diverse neighbor if
   * necessary.
   */
  public void insert(int neighborId, float score, float alpha) throws IOException {
    assert neighborId != nodeId : "can't add self as neighbor at node " + nodeId;
    neighborsRef.getAndUpdate(
        current -> {
          ConcurrentNeighborArray next = current.copy();
          next.insertSorted(neighborId, score);
          enforceMaxConnLimit(next, alpha);
          return next;
        });
  }

  public void insert(int neighborId, float score) throws IOException {
    insert(neighborId, score, 1.0f);
  }

  // is the candidate node with the given score closer to the base node than it is to any of the
  // existing neighbors
  private boolean isDiverse(int node, float score, NeighborArray others, BitSet selected, float alpha) {
    if (others.size() == 0) {
      return true;
    }

    NeighborSimilarity.ScoreFunction scoreProvider = similarity.scoreProvider(node);
    for (int i = others.size() - 1; i >= 0; i--) {
      if (!selected.get(i)) {
        continue;
      }
      int otherNode = others.node[i];
      if (node == otherNode) {
        break;
      }
      if (scoreProvider.apply(otherNode) > score * alpha) {
        return false;
      }
    }
    return true;
  }

  private void enforceMaxConnLimit(NeighborArray neighbors, float alpha) {
    while (neighbors.size() > maxConnections) {
      try {
        removeLeastDiverse(neighbors, alpha);
      } catch (IOException e) {
        throw new UncheckedIOException(e); // called from closures
      }
    }
  }

  /**
   * For each node e1 starting with the last neighbor (i.e. least similar to the base node), look at
   * all nodes e2 that are closer to the base node than e1 is. If any e2 is closer to e1 than e1 is
   * to the base node, remove e1.
   */
  private void removeLeastDiverse(NeighborArray neighbors, float alpha) throws IOException {
    for (int i = neighbors.size() - 1; i >= 1; i--) {
      int e1Id = neighbors.node[i];
      float baseScore = neighbors.score[i];
      NeighborSimilarity.ScoreFunction scoreProvider = similarity.scoreProvider(e1Id);

      for (int j = i - 1; j >= 0; j--) {
        int n2Id = neighbors.node[j];
        float n1n2Score = scoreProvider.apply(n2Id);
        if (n1n2Score > baseScore * alpha) {
          neighbors.removeIndex(i);
          return;
        }
      }
    }

    // couldn't find any "non-diverse" neighbors, so remove the one farthest from the base node
    neighbors.removeIndex(neighbors.size() - 1);
  }

  public ConcurrentNeighborSet copy() {
    return new ConcurrentNeighborSet(this);
  }

  /** Only for testing; this is a linear search */
  boolean contains(int i) {
    var it = this.nodeIterator();
    while (it.hasNext()) {
      if (it.nextInt() == i) {
        return true;
      }
    }
    return false;
  }

  /** Encapsulates comparing node distances for diversity checks. */
  public interface NeighborSimilarity {
    /** for one-off comparisons between nodes */
    float score(int node1, int node2);

    /**
     * For when we're going to compare node1 with multiple other nodes. This allows us to skip
     * loading node1's vector (potentially from disk) redundantly for each comparison.
     */
    ScoreFunction scoreProvider(int node1);

    /**
     * A Function&lt;Integer, Float&gt; without the boxing
     */
    @FunctionalInterface
    public interface ScoreFunction {
      float apply(int node);
    }
  }

  /** A NeighborArray that knows how to copy itself and that checks for duplicate entries */
  static class ConcurrentNeighborArray extends NeighborArray {
    public ConcurrentNeighborArray(int maxSize, boolean descOrder) {
      super(maxSize, descOrder);
    }

    // two nodes may attempt to add each other in the Concurrent classes,
    // so we need to check if the node is already present.  this means that we can't use
    // the parent approach of "append it, and then move it into place"
    @Override
    public void insertSorted(int newNode, float newScore) {
      if (size == node.length) {
        growArrays();
      }
      int insertionPoint =
          scoresDescOrder
              ? descSortFindRightMostInsertionPoint(newScore, size)
              : ascSortFindRightMostInsertionPoint(newScore, size);
      if (!duplicateExistsNear(insertionPoint, newNode, newScore)) {
        System.arraycopy(node, insertionPoint, node, insertionPoint + 1, size - insertionPoint);
        System.arraycopy(score, insertionPoint, score, insertionPoint + 1, size - insertionPoint);
        node[insertionPoint] = newNode;
        score[insertionPoint] = newScore;
        ++size;
      }
    }

    private boolean duplicateExistsNear(int insertionPoint, int newNode, float newScore) {
      // Check to the left
      for (int i = insertionPoint - 1; i >= 0 && score[i] == newScore; i--) {
        if (node[i] == newNode) {
          return true;
        }
      }

      // Check to the right
      for (int i = insertionPoint; i < size && score[i] == newScore; i++) {
        if (node[i] == newNode) {
          return true;
        }
      }

      return false;
    }

    public ConcurrentNeighborArray copy() {
      ConcurrentNeighborArray copy = new ConcurrentNeighborArray(node.length, scoresDescOrder);
      copy.size = size;
      System.arraycopy(node, 0, copy.node, 0, size);
      System.arraycopy(score, 0, copy.score, 0, size);
      return copy;
    }
  }
}
