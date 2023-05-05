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

import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.SparseFixedBitSet;
import org.junit.Before;

import java.io.IOException;
import java.util.HashSet;

public class TestHnswGraphResumableSearcher extends FloatVectorHnswGraphTestCase
    implements SerialHnswGraphTest<float[]> {
  @Before
  @Override
  public void setUp() throws Exception {
    super.setUp();
    this.factory = OnHeapHnswGraphFactory.instance;
  }

  public void testSearchAndResume() throws IOException {
    int nDoc = 1000;
    int topK = 10;

    RandomAccessVectorValues<float[]> vectors = circularVectorValues(nDoc);
    HnswGraphBuilder<float[]> builder =
            factory.createBuilder(
                    vectors, getVectorEncoding(), similarityFunction, 10, 100, random().nextInt());
    HnswGraph hnsw = builder.build(vectors.copy());

    NeighborQueue searchSpace = new NeighborQueue(topK, true);
    BitSet evaluated = new SparseFixedBitSet(nDoc);
    BitSet seen = new SparseFixedBitSet(nDoc);

    float[] queryVector = vectors.vectorValue(0);
    HnswGraphResumableSearcher<float[]> resumableSearcher = new HnswGraphResumableSearcher<>(
            queryVector,
            vectors,
            searchSpace,
            getVectorEncoding(),
            similarityFunction,
            hnsw,
            null,
            evaluated,
            seen);

    // simple test that we get the same topK with a vanilla search
    NeighborQueue classicResults = HnswGraphSearcher.search(queryVector, topK, vectors, getVectorEncoding(), similarityFunction, hnsw, null, Integer.MAX_VALUE);
    NeighborQueue searchResults = resumableSearcher.search(topK, Integer.MAX_VALUE);
    assertResultsEqual(classicResults, searchResults);

    // Search + resume we should see the same top 2*k as a search(2*k), but not necessarily in the same order
    var L1 = new HashSet<Integer>();
    // (re-start the search since assertResults pops them off)
    searchResults = resumableSearcher.search(topK, Integer.MAX_VALUE);
    while (searchResults.size() > 0) {
        L1.add(searchResults.pop());
    }
    searchResults = resumableSearcher.resume(topK, Integer.MAX_VALUE);
    while (searchResults.size() > 0) {
        L1.add(searchResults.pop());
    }

    classicResults = HnswGraphSearcher.search(queryVector, 2*topK, vectors, getVectorEncoding(), similarityFunction, hnsw, null, Integer.MAX_VALUE);
    var L2 = new HashSet<Integer>();
    while (classicResults.size() > 0) {
        L2.add(classicResults.pop());
    }
    assertEquals(L2, L1);
  }
}
