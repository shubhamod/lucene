package org.apache.lucene.util.hnsw;

import org.apache.lucene.tests.util.LuceneTestCase;

public class TestFixedNeighborArray extends LuceneTestCase {
  public void testPushWhenArrayNotFull() {
    int maxSize = 5;
    FixedNeighborArray fna = new FixedNeighborArray(maxSize);
    fna.push(1, 0.5f);
    fna.push(2, 0.6f);
    assertEquals(2, fna.size);
    assertEquals(1, fna.node[1]);
    assertEquals(2, fna.node[0]);
    assertEquals(0.5f, fna.score[1], 1e-6);
    assertEquals(0.6f, fna.score[0], 1e-6);
  }

  public void testPushWhenArrayFullAndScoreLower() {
    int maxSize = 3;
    FixedNeighborArray fna = new FixedNeighborArray(maxSize);
    fna.push(1, 0.7f);
    fna.push(2, 0.6f);
    fna.push(3, 0.8f);
    fna.push(4, 0.5f);
    assertEquals(3, fna.size);
    assertEquals(2, fna.node[2]);
    assertEquals(1, fna.node[1]);
    assertEquals(3, fna.node[0]);
    assertEquals(0.6f, fna.score[2], 1e-6);
    assertEquals(0.7f, fna.score[1], 1e-6);
    assertEquals(0.8f, fna.score[0], 1e-6);
  }

  public void testPushWhenArrayFullAndScoreHigher() {
    int maxSize = 3;
    FixedNeighborArray fna = new FixedNeighborArray(maxSize);
    fna.push(1, 0.7f);
    fna.push(2, 0.6f);
    fna.push(3, 0.8f);
    fna.push(4, 0.9f);
    assertEquals(3, fna.size);
    assertEquals(1, fna.node[2]);
    assertEquals(3, fna.node[1]);
    assertEquals(4, fna.node[0]);
    assertEquals(0.7f, fna.score[2], 1e-6);
    assertEquals(0.8f, fna.score[1], 1e-6);
    assertEquals(0.9f, fna.score[0], 1e-6);
  }

  public void testRandom() {
    int maxSize = random().nextInt(10);
    FixedNeighborArray fna = new FixedNeighborArray(maxSize);
    float maxGenerated = 0;
    int maxOrdinal = 0;
    for (int i = 0; i < 10000; i++) {
      float a = random().nextFloat();
      if (a > maxGenerated) {
        maxGenerated = a;
        maxOrdinal = i;
      }
      fna.push(i, a);

      assertEquals(maxGenerated, fna.score[0], 1e-6);
      assertEquals(maxOrdinal, fna.node[0]);
      assertEquals(Math.min(i + 1, maxSize), fna.size);
      // scores are sorted highest to lowest
      for (int j = 0; j < fna.size - 1; j++) {
        assertTrue(fna.score[j] >= fna.score[j + 1]);
      }
    }
  }
}
