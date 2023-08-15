package org.apache.lucene.util.hnsw;

import org.apache.lucene.tests.util.LuceneTestCase;

import java.util.HashSet;
import java.util.Set;

public class TestSparseIntSet extends LuceneTestCase {
  public void testRandom() {
    SparseIntSet set = new SparseIntSet();
    addItems(set);
    set.clear();
    addItems(set);
  }

  private static void addItems(SparseIntSet set) {
    Set<Integer> expected = new HashSet<>();
    for (int i = 0; i < 2000; i++) {
      int value = random().nextInt(1000);
      if (!set.contains(value)) {
        set.add(value);
      }
      expected.add(value);
    }

    assertEquals(expected.size(), set.size());
    for (int i = 0; i < 1000; i++) {
      assertEquals(expected.contains(i), set.contains(i));
    }
  }
}
