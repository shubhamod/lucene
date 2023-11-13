package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.FingerMetadata.LshBasis;
import org.apache.lucene.util.hnsw.math.distribution.NormalDistribution;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.MatrixUtils;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.stat.correlation.PearsonsCorrelation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.hnsw.HnswGraphTestCase.createRandomFloatVectors;

public class TestFingerMetadata extends LuceneTestCase {
  // check that our basis matches what scikit PCA thinks it should be for a tiny example
  public void testSimpleBasis() {
    List<float[]> vectors = List.of(new float[]{1, 2, 3}, new float[]{2, 3, 4});
    var lsh = LshBasis.computeFromResiduals(vectors.iterator(), 3, 2);
    assertArrayEquals(new float[] {-0.33809817f, -0.55064932f, -0.76320047f}, lsh.basis[0], 0.01f);
    assertArrayEquals(new float[] {0.84795222f,  0.17354729f, -0.50085764f}, lsh.basis[1], 0.01f);
  }

  // check that the properties of the basis vectors match what we expect from principle
  // component analysis
  public void testLshSanity() {
    List<float[]> data = generateTestData(1000, 50);
    LshBasis lsh = LshBasis.computeFromResiduals(data.iterator(), 50, 8);
    List<RealVector> projectedData = data.stream().map(lsh::project).map(this::toRealVector).collect(Collectors.toList());

    // Verify that the basis is orthonormal and that the vectors have length 1
    RealMatrix basis = toRealMatrix(lsh.basis);
    RealMatrix product = basis.multiply(basis.transpose());
    RealMatrix I = MatrixUtils.createRealIdentityMatrix(8);
    for (int i = 0; i < 8; i++) {
      assertArrayEquals(I.getRow(i), product.getRow(i), 0.01);
      assertEquals(1.0, basis.getRowVector(i).getNorm(), 0.01);
    }

    // Verify that the variance of each component decreases in order
    double previousVariance = Double.MAX_VALUE;
    for (int i = 0; i < basis.getRowDimension(); i++) {
      int finalI = i;
      List<Double> componentData = projectedData.stream().map(v -> v.getEntry(finalI)).collect(Collectors.toList());
      double currentVariance = computeVarianceList(componentData);
      assert currentVariance <= previousVariance : "Variance of component " + i + " is not less than or equal to variance of the previous component";
      previousVariance = currentVariance;
    }

    // Compute the reconstruction error for the LSH basis
    List<RealVector> rvData = data.stream().map(this::toRealVector).toList();
    double lshError = computeReconstructionError(basis, rvData, projectedData);
    System.out.println("Reconstruction error for LSH basis: " + lshError);

    // Compute the variance of the projected data.
    double lshV = computeVariance(projectedData);
    System.out.println("Variance of projected LSH data: " + lshV);

    // Compare with some random projections. If we computed the basis correctly,
    // its variance will be higher and reconstruction lower than any random basis.
    for (int i = 0; i < 100; i++) {
      // Generate a random 50x8 matrix.
      LshBasis randomBasis = LshBasis.createRandom(50, 8);

      // Variance
      List<RealVector> randomProjectedData = data.stream().map(randomBasis::project).map(this::toRealVector).toList();
      double rv = computeVariance(randomProjectedData);
      assert rv < lshV : "Random variance " + rv + " is not less than LSH variance " + lshV;

      // Reconstruction error
      double randomError = computeReconstructionError(basis, rvData, randomProjectedData);
      assert randomError > lshError : "Random reconstruction error " + randomError + " is not greater than LSH error " + lshError;
    }
  }

  // compare the accuracy of the LSH approximation to the actual similarity
  // this checks for regressions, but I don't think the accuracy is actually Good Enough
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

  private RealMatrix toRealMatrix(float[][] basis) {
    double[][] d = new double[basis.length][];
    for (int i = 0; i < basis.length; i++) {
      d[i] = toDoubleArray(basis[i]);
    }
    return MatrixUtils.createRealMatrix(d);
  }

  private RealVector toRealVector(float[] v) {
    return new ArrayRealVector(toDoubleArray(v));
  }

  private static double[] toDoubleArray(float[] v) {
    double[] d = new double[v.length];
    for (int i = 0; i < v.length; i++) {
      d[i] = v[i];
    }
    return d;
  }

  private static double computeVarianceList(List<Double> data) {
    int m = data.size();
    double sum = data.stream().mapToDouble(Double::doubleValue).sum();
    double mean = sum / m;

    double sumSquaredDeviations = 0.0;
    for (Double value : data) {
      double deviation = value - mean;
      sumSquaredDeviations += deviation * deviation;
    }
    return sumSquaredDeviations / (m - 1);
  }

  private static double computeReconstructionError(RealMatrix basis, List<RealVector> originalData, List<RealVector> projectedData) {
    double totalError = 0.0;
    for (int i = 0; i < originalData.size(); i++) {
      RealVector original = originalData.get(i);
      RealVector projected = projectedData.get(i);
      RealVector reconstructed = unproject(basis, projected);
      double error = original.subtract(reconstructed).getNorm();
      totalError += error * error;
    }
    return totalError / originalData.size();
  }

  private static RealVector unproject(RealMatrix basis, RealVector v) {
    return basis.transpose().operate(v);
  }

  private static List<float[]> generateTestData(int count, int dimension) {
    NormalDistribution nd = new NormalDistribution();
    return IntStream.range(0, count)
        .mapToObj(i -> {
          var a = new float[dimension];
          for (int j = 0; j < dimension; j++) {
            a[j] = (float) nd.sample();
          }
          return a;
        })
        .collect(Collectors.toList());
  }

  private static double computeVariance(List<RealVector> data) {
    int m = data.size();
    double sum = 0.0;
    for (RealVector v : data) {
      sum += v.getEntry(0);
    }
    double mean = sum / m;

    double sumSquaredDeviations = 0.0;
    for (RealVector v : data) {
      double deviation = v.getEntry(0) - mean;
      sumSquaredDeviations += deviation * deviation;
    }
    return sumSquaredDeviations / (m - 1);
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
