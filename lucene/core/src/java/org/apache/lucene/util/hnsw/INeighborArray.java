package org.apache.lucene.util.hnsw;

/** abstracts efficient neighbor arrays */
public interface INeighborArray {
  int size();
  int[] node();
  float[] score();
  boolean scoresDescending();
}
