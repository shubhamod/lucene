package org.apache.lucene.util.hnsw;

import java.io.IOException;
import java.util.ArrayList;

public class ArrayListVectorValues<T> implements RandomAccessVectorValues<T> {
  private final ArrayList<T> values;
  private final int dimension;

  public ArrayListVectorValues(int dimension) {
    this(new ArrayList<>(), dimension);
  }

  public ArrayListVectorValues(ArrayList<T> values, int dimension) {
    this.values = values;
    this.dimension = dimension;
  }

  @Override
  public int size() {
    return values.size();
  }

  @Override
  public int dimension() {
    return dimension;
  }

    @Override
    public T vectorValue(int targetOrd) {
        return values.get(targetOrd);
    }

    @Override
    public RandomAccessVectorValues<T> copy() throws IOException {
      // shwllow copy is fine
      return new ArrayListVectorValues<>(values, dimension);
    }
}
