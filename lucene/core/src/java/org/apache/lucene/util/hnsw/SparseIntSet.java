package org.apache.lucene.util.hnsw;

import java.util.Arrays;

/** A Set specialized for ints */
public class SparseIntSet {
  private static final int DEFAULT_INITIAL_CAPACITY = 16;
  private static final float DEFAULT_LOAD_FACTOR = 0.75f;

  private int[] table;
  private boolean[] occupied;
  private int size;
  private int threshold;

  public SparseIntSet() {
    table = new int[DEFAULT_INITIAL_CAPACITY];
    occupied = new boolean[DEFAULT_INITIAL_CAPACITY];
    threshold = (int) (DEFAULT_INITIAL_CAPACITY * DEFAULT_LOAD_FACTOR);
  }

  public boolean contains(int value) {
    int index = indexFor(value, table.length);
    while (occupied[index]) {
      if (table[index] == value) {
        return true;
      }
      index = (index + 1) % table.length;
    }
    return false;
  }

  public void add(int value) {
    if (size >= threshold) {
      resize();
    }

    int index = indexFor(value, table.length);
    while (occupied[index]) {
      if (table[index] == value) {
        return; // value is already present
      }
      index = (index + 1) % table.length;
    }

    occupied[index] = true;
    table[index] = value;
    size++;
  }

  public void clear() {
    Arrays.fill(occupied, false);
    size = 0;
  }

  private void resize() {
    int oldCapacity = table.length;
    int newCapacity = oldCapacity * 2;
    int[] newTable = new int[newCapacity];
    boolean[] newOccupied = new boolean[newCapacity];

    for (int i = 0; i < oldCapacity; i++) {
      if (occupied[i]) {
        int value = table[i];
        int index = indexFor(value, newCapacity);
        while (newOccupied[index]) {
          index = (index + 1) % newCapacity;
        }
        newOccupied[index] = true;
        newTable[index] = value;
      }
    }

    table = newTable;
    occupied = newOccupied;
    threshold = (int) (newCapacity * DEFAULT_LOAD_FACTOR);
  }

  private int indexFor(int value, int capacity) {
    return (value & 0x7fffffff) % capacity; // Ensure it's positive
  }

  public int size() {
    return size;
  }
}
