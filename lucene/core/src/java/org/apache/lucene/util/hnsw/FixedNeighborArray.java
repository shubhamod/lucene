package org.apache.lucene.util.hnsw;

import org.apache.lucene.util.BitSet;

import java.util.Arrays;
import java.util.HashSet;

/**
 * Stores nodes and scores in descending order.  The best scores are highest and will
 * be at the start of the array.
 */
public class FixedNeighborArray implements INeighborArray {
  int size;
  int cur;

  float[] score;
  int[] node;

  HashSet<Integer> contents;

  public FixedNeighborArray(int maxSize) {
    node = new int[maxSize];
    score = new float[maxSize];
    contents = new HashSet<>(maxSize);
  }

  @Override
  public int size() {
    return size;
  }

  @Override
  public int[] node() {
    return node;
  }

  @Override
  public float[] score() {
    return score;
  }

  @Override
  public boolean scoresDescending() {
    return true;
  }

  public boolean contains(int node) {
    return contents.contains(node);
  }

  /**
   * Adds a new node to the NeighborArray if there is room or if the new node is better than the
   * worst node currently in the array.
   */
  public void push(int newNode, float newScore) {
    if (size == node.length && newScore <= score[size - 1]) {
      return;
    }

    // find the insertion point with binary search
    int lo = 0;
    int hi = size;
    while (lo < hi) {
      int mid = (lo + hi) / 2;
      if (newScore > score[mid]) {
        hi = mid;
      } else if (newScore < score[mid]) {
        lo = mid + 1;
      } else {
        // equal scores
        lo = mid;
        break;
      }
    }
    assert lo < node.length; // we checked at the start for score being good enough

    // move existing nodes down to make room for the new node
    if (lo < size) {
      int numToMove = size < node.length ? size - lo : size - lo - 1;
      System.arraycopy(node, lo, node, lo + 1, numToMove);
      System.arraycopy(score, lo, score, lo + 1, numToMove);
    }

    // insert the new node
    node[lo] = newNode;
    score[lo] = newScore;
    if (size < node.length) {
      ++size;
    }

    if (lo < cur) {
      cur = lo;
    }

    contents.add(newNode);
  }

  int nextUnvisited(BitSet visited) {
    for ( ; cur < size; cur++) {
      int n = this.node[cur];
      if (!visited.get(n)) {
        return cur++;
      }
    }
    return -1;
  }

  // TODO add topK parameter? so we only copy out what the caller wants
  public int[] nodesCopy() {
    return Arrays.copyOf(node, size);
  }

  public float[] scoresCopy() {
    return Arrays.copyOf(score, size);
  }

  public void clear() {
    size = 0;
    cur = 0;
  }
}
