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
import static org.apache.lucene.util.StringHelper.ID_LENGTH;

import com.carrotsearch.randomizedtesting.RandomizedTest;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.lucene95.Lucene95Codec;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsFormat;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsReader;
import org.apache.lucene.codecs.lucene95.Lucene95HnswVectorsWriter;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.ArrayUtil;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.junit.Before;

/** Tests HNSW KNN graphs */
public class TestConcurrentHnswFloatVectorGraph extends ConcurrentHnswGraphTestCase<float[]> {

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
    VectorEncoding vectorEncoding = getVectorEncoding();
    random().nextInt();
    ConcurrentHnswGraphBuilder<float[]> builder =
        new ConcurrentHnswGraphBuilder<>(vectors, vectorEncoding, similarityFunction, 16, 100);
    ConcurrentOnHeapHnswGraph hnsw = buildParallel(builder, vectors);

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
            hnsw.getView(),
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

  public void testWriteRead() throws IOException {
    similarityFunction = VectorSimilarityFunction.EUCLIDEAN;
    // todo more vectors and make sure that we have the same graph after writing and reading
    var rawVectors = List.of(new float[] {1.1f, 2.2f},
            new float[] {3.3f, 4.4f},
            new float[] {5.5f, 6.6f},
            new float[] {7.7f, 8.8f},
            new float[] {9.9f, 10.10f});
    RandomAccessVectorValues<float[]> vectors = new Lucene95HnswVectorsWriter.RAVectorValues<>(rawVectors, 2);
    VectorEncoding vectorEncoding = getVectorEncoding();
    random().nextInt();
    ConcurrentHnswGraphBuilder<float[]> builder =
            new ConcurrentHnswGraphBuilder<>(vectors, vectorEncoding, similarityFunction, 16, 100);
    ConcurrentOnHeapHnswGraph hnsw = buildParallel(builder, vectors);

    Path vectorPath = LuceneTestCase.createTempFile();
    var segmentId = writeGraph(vectorPath, rawVectors, hnsw);
    try (var reader = openReader(vectorPath, similarityFunction, rawVectors.size(), segmentId)) {
      var graph = reader.getGraph("MockName");
      assertGraphEqual(hnsw.getView(), graph);
      var topDocs = reader.search("MockName", new float[] {8.1f, 9.2f}, 100, null, Integer.MAX_VALUE);
      for (var scoreDoc : topDocs.scoreDocs) {
          System.out.println(scoreDoc.doc);
      }
    }
  }

  private static KnnVectorsReader openReader(Path vectorPath, VectorSimilarityFunction similarityFunction, int size, byte[] segmentId) throws IOException {
    Directory directory = FSDirectory.open(vectorPath.getParent());

    FieldInfo fieldInfo = createFieldInfoForVector(similarityFunction, 2);
    FieldInfos fieldInfos = new FieldInfos(Collections.singletonList(fieldInfo).toArray(new FieldInfo[0]));
    String segmentName = vectorPath.getFileName().toString();
    SegmentInfo segmentInfo = new SegmentInfo(directory, Version.LATEST, Version.LATEST, segmentName, size, false, Lucene95Codec.getDefault(), Collections.emptyMap(), segmentId, Collections.emptyMap(), null);

    SegmentReadState state = new SegmentReadState(directory, segmentInfo, fieldInfos, IOContext.DEFAULT);
    return new Lucene95HnswVectorsFormat(16, 100).fieldsReader(state);
  }

  private byte[] writeGraph(Path vectorPath, List<float[]> rawVectors, ConcurrentOnHeapHnswGraph hnsw) throws IOException {
    // table directory
    Directory directory = FSDirectory.open(vectorPath.getParent());
    // segment name in SAI naming pattern, e.g. ca-3g5d_1t56_18d8122br2d3mg6twm-bti-SAI+ba+table_00_val_idx+Vector.db
    String segmentName = vectorPath.getFileName().toString();

    var segmentId = new byte[ID_LENGTH]; // I don't want to deal with reading and writing this
    SegmentInfo segmentInfo = new SegmentInfo(directory, Version.LATEST, Version.LATEST, segmentName, -1, false, Lucene95Codec.getDefault(), Collections.emptyMap(), segmentId, Collections.emptyMap(), null);
    SegmentWriteState state = new SegmentWriteState(InfoStream.getDefault(), directory, segmentInfo, null, null, IOContext.DEFAULT);
    var fieldInfo = createFieldInfoForVector(similarityFunction, rawVectors.get(0).length);
    try (var writer = new Lucene95HnswVectorsWriter(hnsw, rawVectors, state, 16, 100)) {
      var fw = writer.addField(fieldInfo);
      for (var i = 0; i < rawVectors.size(); i++) {
        fw.addValueForExistingGraph(i, rawVectors.get(i));
      }
      writer.flush(hnsw.size(), null);
      writer.finish();
    }
    return segmentId;
  }

  private static FieldInfo createFieldInfoForVector(VectorSimilarityFunction similarityFunction, int dimension)
  {
    String name = "MockName";
    int number = 0;
    boolean storeTermVector = false;
    boolean omitNorms = false;
    boolean storePayloads = false;
    IndexOptions indexOptions = IndexOptions.NONE;
    DocValuesType docValues = DocValuesType.NONE;
    long dvGen = -1;
    Map<String, String> attributes = Map.of();
    int pointDimensionCount = 0;
    int pointIndexDimensionCount = 0;
    int pointNumBytes = 0;
    VectorEncoding vectorEncoding = VectorEncoding.FLOAT32;
    boolean softDeletesField = false;

    return new FieldInfo(name, number, storeTermVector, omitNorms, storePayloads, indexOptions, docValues,
            dvGen, attributes, pointDimensionCount, pointIndexDimensionCount, pointNumBytes,
            dimension, vectorEncoding, similarityFunction, softDeletesField);
  }

}
