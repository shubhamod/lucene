package org.apache.lucene.util.hnsw;

import org.apache.lucene.tests.util.LuceneTestCase;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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

  public void testVisited() {
    int maxSize = 5;
    FixedNeighborArray fna = new FixedNeighborArray(maxSize);

    fna.push(1, 0.5f);
    assertEquals(0, fna.nextUnvisited());
    fna.push(2, 0.4f);
    assertEquals(1, fna.nextUnvisited());
    assertEquals(-1, fna.nextUnvisited());

    fna.push(3, 0.9f);
    assertEquals(0, fna.nextUnvisited()); // It should revisit this index due to the new highest score
    fna.push(4, 0.7f);
    assertEquals(1, fna.nextUnvisited());
    fna.push(5, 0.8f);
    assertEquals(1, fna.nextUnvisited());
    assertEquals(-1, fna.nextUnvisited());
  }

  public void testRandom() {
    int maxSize = 2 + random().nextInt(10);
    FixedNeighborArray fna = new FixedNeighborArray(maxSize);
    float maxGenerated = 0;
    int maxOrdinal = 0;
    for (int i = 0; i < 10000; i++) {
      if (random().nextFloat() < 0.1f && fna.size() > 0) {
        int j = random().nextInt(fna.size());
        int node = fna.node()[j];
        if (!fna.alreadyEvaluated(node)) {
          fna.push(node, fna.score()[j]);
        }
        continue;
      }

      float a = random().nextFloat();
      if (a > maxGenerated) {
        maxGenerated = a;
        maxOrdinal = i;
      }
      fna.push(i, a);

      assert fna.size() <= maxSize;
      assertEquals(maxGenerated, fna.score[0], 1e-6);
      assertEquals(maxOrdinal, fna.node[0]);
      // scores are sorted highest to lowest
      for (int j = 0; j < fna.size - 1; j++) {
        assertTrue(fna.score[j] >= fna.score[j + 1]);
      }
    }
  }

  public void testVisitedRandom() {
    int maxSize = 2 + random().nextInt(10);
    FixedNeighborArray fna = new FixedNeighborArray(maxSize);
    List<Float> expectedScores = new ArrayList<>();
    Set<Integer> visitedNodes = new HashSet<>();

    for (int i = 0; i < 10000; i++) { // Arbitrary number of iterations
      if (random().nextFloat() < 0.6 || fna.size() == 0) { // 60% chance to push, or if array is empty
        float score = random().nextFloat();
        fna.push(i, score);
        expectedScores.add(score);
        expectedScores.sort(Collections.reverseOrder()); // Keep it sorted in descending order
        if (expectedScores.size() > maxSize) {
          expectedScores.remove(expectedScores.size() - 1); // Remove the smallest score if we exceed maxSize
        }
      } else {
        checkNextVisited(fna, expectedScores, visitedNodes);
      }
    }

    while (checkNextVisited(fna, expectedScores, visitedNodes) >= 0) {
    }
    // Ensure all indices have been visited after all operations
    for (int i = 0; i < fna.size(); i++) {
      assertTrue(visitedNodes.contains(fna.node()[i]));
    }
  }

  private static int checkNextVisited(FixedNeighborArray fna, List<Float> expectedScores, Set<Integer> visitedNodes) {
    int idx = fna.nextUnvisited();
    if (idx >= 0) {
      int node = fna.node()[idx];
      assertFalse("Index already visited!", visitedNodes.contains(node));
      assertEquals("Unexpected score at index!", expectedScores.get(idx), fna.score()[idx], 1e-6);
      visitedNodes.add(node);
    }
    return idx;
  }
}
