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

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

import com.carrotsearch.randomizedtesting.RandomizedTest;
import java.io.IOException;
import java.util.HashSet;

import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.SparseFixedBitSet;
import org.junit.Before;

/** Tests HNSW KNN graphs */
public class TestHnswFloatVectorGraph extends HnswGraphTestCase<float[]> {

  @Before
  public void setup() {
    similarityFunction = RandomizedTest.randomFrom(VectorSimilarityFunction.values());
  }

  @Override
  VectorEncoding getVectorEncoding() {
    return VectorEncoding.FLOAT32;
  }

  @Override
  Query knnQuery(String field, float[] vector, int k) {
    return new KnnFloatVectorQuery(field, vector, k);
  }

  @Override
  float[] randomVector(int dim) {
    return randomVector(random(), dim);
  }

  @Override
  AbstractMockVectorValues<float[]> vectorValues(int size, int dimension) {
    return MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, random()));
  }

  @Override
  AbstractMockVectorValues<float[]> vectorValues(float[][] values) {
    return MockVectorValues.fromValues(values);
  }

  @Override
  AbstractMockVectorValues<float[]> vectorValues(LeafReader reader, String fieldName)
      throws IOException {
    FloatVectorValues vectorValues = reader.getFloatVectorValues(fieldName);
    float[][] vectors = new float[reader.maxDoc()][];
    while (vectorValues.nextDoc() != NO_MORE_DOCS) {
      vectors[vectorValues.docID()] =
          ArrayUtil.copyOfSubArray(
              vectorValues.vectorValue(), 0, vectorValues.vectorValue().length);
    }
    return MockVectorValues.fromValues(vectors);
  }

  @Override
  AbstractMockVectorValues<float[]> vectorValues(
      int size,
      int dimension,
      AbstractMockVectorValues<float[]> pregeneratedVectorValues,
      int pregeneratedOffset) {
    float[][] vectors = new float[size][];
    float[][] randomVectors =
        createRandomFloatVectors(
            size - pregeneratedVectorValues.values.length, dimension, random());

    for (int i = 0; i < pregeneratedOffset; i++) {
      vectors[i] = randomVectors[i];
    }

    int currentDoc;
    while ((currentDoc = pregeneratedVectorValues.nextDoc()) != NO_MORE_DOCS) {
      vectors[pregeneratedOffset + currentDoc] = pregeneratedVectorValues.values[currentDoc];
    }

    for (int i = pregeneratedOffset + pregeneratedVectorValues.values.length;
        i < vectors.length;
        i++) {
      vectors[i] = randomVectors[i - pregeneratedVectorValues.values.length];
    }

    return MockVectorValues.fromValues(vectors);
  }

  @Override
  Field knnVectorField(String name, float[] vector, VectorSimilarityFunction similarityFunction) {
    return new KnnFloatVectorField(name, vector, similarityFunction);
  }

  @Override
  RandomAccessVectorValues<float[]> circularVectorValues(int nDoc) {
    return new CircularFloatVectorValues(nDoc);
  }

  @Override
  float[] getTargetVector() {
    return new float[] {1f, 0f};
  }

  public void testSearchWithSkewedAcceptOrds() throws IOException {
    int nDoc = 1000;
    similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
    RandomAccessVectorValues<float[]> vectors = circularVectorValues(nDoc);
    HnswGraphBuilder<float[]> builder =
        HnswGraphBuilder.create(
            vectors, getVectorEncoding(), similarityFunction, 16, 100, random().nextInt());
    OnHeapHnswGraph hnsw = builder.build(vectors.copy());

    // Skip over half of the documents that are closest to the query vector
    FixedBitSet acceptOrds = new FixedBitSet(nDoc);
    for (int i = 500; i < nDoc; i++) {
      acceptOrds.set(i);
    }
    NeighborQueue nn =
        HnswGraphSearcher.search(
            getTargetVector(),
            10,
            vectors.copy(),
            getVectorEncoding(),
            similarityFunction,
            hnsw,
            acceptOrds,
            Integer.MAX_VALUE);

    int[] nodes = nn.nodes();
    assertEquals("Number of found results is not equal to [10].", 10, nodes.length);
    int sum = 0;
    for (int node : nodes) {
      assertTrue("the results include a deleted document: " + node, acceptOrds.get(node));
      sum += node;
    }
    // We still expect to get reasonable recall. The lowest non-skipped docIds
    // are closest to the query vector: sum(500,509) = 5045
    assertTrue("sum(result docs)=" + sum, sum < 5100);
  }

  public void testEmptyGraph() throws IOException {
    HnswGraph hnsw = new OnHeapHnswGraph(1);
    var vectors = emptyFloatValues(10);
    float[] query = randomVector(10);
    var topK = 1;
    HnswGraphResumableSearcher<float[]> resumableSearcher =
        new HnswGraphResumableSearcher<>(
            query,
            vectors,
            new NeighborQueue(topK, true),
            getVectorEncoding(),
            similarityFunction,
            hnsw,
            null,
            new SparseFixedBitSet(1),
            new SparseFixedBitSet(1));

    NeighborQueue searchResults = resumableSearcher.search(topK, Integer.MAX_VALUE);
    assertEquals(0, searchResults.size());
    searchResults = resumableSearcher.resume(topK, Integer.MAX_VALUE);
    assertEquals(0, searchResults.size());
  }

  public void testSingleNodeGraph() throws IOException {
    var vectors = circularVectorValues(1);
    var builder = HnswGraphBuilder.create(
            vectors, getVectorEncoding(), similarityFunction, 10, 100, random().nextInt());
    HnswGraph hnsw = builder.build(vectors.copy());

    float[] query = randomVector(2);
    var topK = 1;
    HnswGraphResumableSearcher<float[]> resumableSearcher =
        new HnswGraphResumableSearcher<>(
            query,
            vectors,
            new NeighborQueue(topK, true),
            getVectorEncoding(),
            similarityFunction,
            hnsw,
            null,
            new SparseFixedBitSet(1),
            new SparseFixedBitSet(1));

    NeighborQueue searchResults = resumableSearcher.search(topK, Integer.MAX_VALUE);
    assertEquals(1, searchResults.size());
    assertArrayEquals(
        vectors.vectorValue(0), vectors.vectorValue(searchResults.topNode()), 0.000001f);
    searchResults = resumableSearcher.resume(topK, Integer.MAX_VALUE);
    assertEquals(0, searchResults.size());
  }

  public void testSearchAndResume() throws IOException {
    int nDoc = atLeast(1000);
    int topK = atLeast(10);

    RandomAccessVectorValues<float[]> vectors = circularVectorValues(nDoc);
    var builder =
        HnswGraphBuilder.create(
            vectors,
            getVectorEncoding(),
            similarityFunction,
            atLeast(8),
            atLeast(50),
            random().nextInt());
    HnswGraph hnsw = builder.build(vectors.copy());

    float[] queryVector = vectors.vectorValue(0);
    HnswGraphResumableSearcher<float[]> resumableSearcher =
        new HnswGraphResumableSearcher<>(
            queryVector,
            vectors,
            new NeighborQueue(topK, true),
            getVectorEncoding(),
            similarityFunction,
            hnsw,
            null,
            new SparseFixedBitSet(nDoc),
            new SparseFixedBitSet(nDoc));

    // simple test that we get the same topK with a vanilla search
    NeighborQueue classicResults =
        HnswGraphSearcher.search(
            queryVector,
            topK,
            vectors,
            getVectorEncoding(),
            similarityFunction,
            hnsw,
            null,
            Integer.MAX_VALUE);
    NeighborQueue searchResults = resumableSearcher.search(topK, Integer.MAX_VALUE);
    assertResultsEqual(classicResults, searchResults);

    // Search + resume we should see the same top 2*k as a search(2*k), but not necessarily in the
    // same order
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

    classicResults =
        HnswGraphSearcher.search(
            queryVector,
            2 * topK,
            vectors,
            getVectorEncoding(),
            similarityFunction,
            hnsw,
            null,
            Integer.MAX_VALUE);
    var L2 = new HashSet<Integer>();
    while (classicResults.size() > 0) {
      L2.add(classicResults.pop());
    }
    assertEquals(L2, L1);
  }
}
