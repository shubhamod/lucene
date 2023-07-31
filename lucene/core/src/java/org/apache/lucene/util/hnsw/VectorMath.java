package org.apache.lucene.util.hnsw;

import org.apache.lucene.util.VectorUtil;

import java.util.Iterator;

public class VectorMath {
  static float[] mapMultiply(float[] a, float f) {
    float[] result = new float[a.length];
    for (int i = 0; i < a.length; i++) {
      result[i] = a[i] * f;
    }
    return result;
  }

  static float[] subtract(float[] a, float[] b) {
      assert a.length == b.length;

      float[] result = new float[a.length];
      for (int i = 0; i < a.length; i++) {
          result[i] = a[i] - b[i];
      }
      return result;
  }

  static void computeOuterProduct(float[] a, float[] b, double[][] result) {
      int dimension = a.length;
      for (int i = 0; i < dimension; i++) {
          for (int j = 0; j < dimension; j++) {
              result[i][j] = a[i] * b[j];
          }
      }
  }

  static void addInPlace(double[][] m1, double[][] m2) {
      int rows = m1.length;
      int cols = m1[0].length;
      for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              m1[i][j] += m2[i][j];
          }
      }
  }

  static void multiplyInPlace(double[][] m, float f) {
      int rows = m.length;
      int cols = m[0].length;
      for (int i = 0; i < rows; i++) {
          for (int j = 0; j < cols; j++) {
              m[i][j] *= f;
          }
      }
  }

  // might as well use double internally since that's what Commons Math wants
  // to compute eigenvectors later, anyway
  static double[][] incrementalCovariance(Iterator<float[]> data, int dimension) {
      if (!data.hasNext()) {
          throw new IllegalArgumentException("No data provided.");
      }

      // we don't center our data -- it's already centered-ish by construction, and
      // doing an extra transform is more work that we don't want to do for each query

      // Initialize sum of squares matrix and outerProduct scratch space
      double[][] sumOfSquares = new double[dimension][dimension];
      double[][] outerProduct = new double[dimension][dimension];

      // Iterate over data
      int count = 0;
      while (data.hasNext()) {
          float[] vector = data.next();
          computeOuterProduct(vector, vector, outerProduct);
          addInPlace(sumOfSquares, outerProduct);
          count++;
      }

      // Compute raw covariance matrix
      multiplyInPlace(sumOfSquares, 1.0f / count);
      return sumOfSquares;
  }

  static double cosine(float[] qRes, float[] dRes) {
      double dotProduct = 0.0;
      double qResNorm = 0.0;
      double dResNorm = 0.0;
      for (int i = 0; i < qRes.length; i++) {
          dotProduct += qRes[i] * dRes[i];
          qResNorm += qRes[i] * qRes[i];
          dResNorm += dRes[i] * dRes[i];
      }
      return dotProduct / (Math.sqrt(qResNorm) * Math.sqrt(dResNorm));
  }

  static float norm(float[] v) {
      return (float) Math.sqrt(VectorUtil.dotProduct(v, v));
  }
}
