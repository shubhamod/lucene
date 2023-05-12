package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentMap;

public class TestConcurrentHnswGraphWhitebox extends LuceneTestCase {
    public void testConcurrentInsert1() throws IOException {
        var vectors = new ArrayListVectorValues<byte[]>(1);
        var maxConn = 16;
        var builder =
                new ConcurrentHnswGraphBuilder<>(vectors, VectorEncoding.BYTE, VectorSimilarityFunction.COSINE, maxConn, 100);
        var g = builder.getGraph().graphLevels;

        // set up
        // 0 -> 1
        // 1 <- 0
        addSkeleton(g, 0, 0, maxConn);
        addSkeleton(g, 1, 0, maxConn);
        g.get(0).get(0).insert(1, 1.0f, this::scoreBetween);
        g.get(0).get(1).insert(0, 1.0f, this::scoreBetween);

        // partially insert 2 and 3
        addSkeleton(g, 3, 1, maxConn);
        builder.insertionsInProgress.add(new ConcurrentOnHeapHnswGraph.NodeAtLevel(1, 3));

    }

    private float scoreBetween(int i, int j) throws IOException {
        return 1.0f;
    }

    private void addSkeleton(Map<Integer, Map<Integer, ConcurrentNeighborSet>> g, int node, int level, int M) {
        for (int i = 0; i < level; i++) {
            var L = g.computeIfAbsent(i, k -> new HashMap<>());
            L.computeIfAbsent(node, k -> new ConcurrentNeighborSet(node, M));
        }
    }
}
