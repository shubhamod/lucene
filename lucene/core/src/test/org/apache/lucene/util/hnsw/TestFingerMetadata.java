package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.FingerMetadata.LshBasis;
import org.apache.lucene.util.hnsw.math.stat.correlation.PearsonsCorrelation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.hnsw.HnswGraphTestCase.createRandomFloatVectors;

public class TestFingerMetadata extends LuceneTestCase {
  public void testSimpleBasis() {
    List<float[]> vectors = List.of(new float[]{1, 2, 3}, new float[]{2, 3, 4});
    var lsh = LshBasis.computeFromResiduals(vectors.iterator(), 3, 2);
    assertArrayEquals(new float[] {-0.33809817f, -0.55064932f, -0.76320047f}, lsh.basis[0], 0.01f);
    assertArrayEquals(new float[] {0.84795222f,  0.17354729f, -0.50085764f}, lsh.basis[1], 0.01f);
  }

  public void testAccuracy() throws IOException {
    var similarityFunction = VectorSimilarityFunction.DOT_PRODUCT;
    var encoding = VectorEncoding.FLOAT32;
    var size = 10000;
    var dimension = 50;
    AbstractMockVectorValues<float[]> vectors = MockVectorValues.fromValues(createRandomFloatVectors(size, dimension, random()));
    HnswGraphBuilder<float[]> builder =
        HnswGraphBuilder.create(
            vectors, encoding, similarityFunction, 10, 30, random().nextLong());
    OnHeapHnswGraph hnsw = builder.build(vectors.copy());

    testAccuracy(similarityFunction, encoding, vectors, hnsw, 4, 0.15);
    testAccuracy(similarityFunction, encoding, vectors, hnsw, 8, 0.10);
    testAccuracy(similarityFunction, encoding, vectors, hnsw, 16, 0.05);
    testAccuracy(similarityFunction, encoding, vectors, hnsw, 32, 0.044);
    testAccuracy(similarityFunction, encoding, vectors, hnsw, 49, 0.039);
  }

  private static void testAccuracy(VectorSimilarityFunction similarityFunction,
                                   VectorEncoding encoding,
                                   AbstractMockVectorValues<float[]> vectors,
                                   OnHeapHnswGraph hnsw,
                                   int lshDimensions,
                                   double expectedMae) throws IOException {
    var qVectors = vectors.copy();
    var a1Vectors = vectors.copy();
    var a2Vectors = vectors.copy();
    FingerMetadata<float[]> fm = new FingerMetadata<>(hnsw, vectors, encoding, similarityFunction, lshDimensions);
    List<Double> estimatedSimilarities = new ArrayList<>();
    List<Double> actualSimilarities = new ArrayList<>();
    double totalAbsoluteError = 0.0;
    int numPairs = 0;
    for (int i = 0; i < hnsw.size(); i++) {
      hnsw.seek(0, i);
      int queryNode = hnsw.nextNeighbor(); // we'll compute correlation around the first neighbor
      float[] q = qVectors.vectorValue(queryNode);
      float[] c = a1Vectors.vectorValue(i);
      var provider = fm.similarityProviderFor(q);
      var nearNode = provider.approximateSimilarityNear(i, VectorUtil.dotProduct(q, c));
      int neighbor;
      while ((neighbor = hnsw.nextNeighbor()) != NO_MORE_DOCS) {
        var estimatedSimilarity = nearNode.apply(neighbor);
        var actualSimilarity = similarityFunction.compare(a1Vectors.vectorValue(queryNode), a2Vectors.vectorValue(neighbor));
        estimatedSimilarities.add((double) estimatedSimilarity);
        actualSimilarities.add((double) actualSimilarity);

        totalAbsoluteError += Math.abs(estimatedSimilarity - actualSimilarity);
        numPairs++;
      }
    }

    double meanAbsoluteError = totalAbsoluteError / numPairs;

    // compute correlation
    double[] estimatedArray = estimatedSimilarities.stream().mapToDouble(Double::doubleValue).toArray();
    writeToFile("est-similarity.txt", estimatedArray);
    double[] actualArray = actualSimilarities.stream().mapToDouble(Double::doubleValue).toArray();
    writeToFile("act-similarity.txt", actualArray);
    PearsonsCorrelation pearsonsCorrelation = new PearsonsCorrelation();
    double correlation = pearsonsCorrelation.correlation(estimatedArray, actualArray);

    System.out.printf("r=%s  correlation: %s, MAE: %s%n", lshDimensions, correlation, meanAbsoluteError);
//    assert meanAbsoluteError < expectedMae : "Mean absolute error " + meanAbsoluteError + " is not less than " + expectedMae;
  }

  private static void writeToFile(String filename, double[] a) {
    try (var writer = new java.io.PrintWriter("/tmp/" + filename)) {
      for (double v : a) {
        writer.println(v);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
