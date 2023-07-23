package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.hnsw.math.linear.ArrayRealVector;
import org.apache.lucene.util.hnsw.math.linear.EigenDecomposition;
import org.apache.lucene.util.hnsw.math.linear.RealMatrix;
import org.apache.lucene.util.hnsw.math.linear.RealVector;
import org.apache.lucene.util.hnsw.math.stat.correlation.PearsonsCorrelation;
import org.apache.lucene.util.hnsw.math.stat.descriptive.moment.VectorialMean;

import java.io.IOException;
import java.util.function.Function;

public class FingerSearcher<T> {
    private final RandomAccessVectorValues<T> values;
    private final VectorEncoding vectorEncoding;
    private final HnswGraph hnswGraph;
    private final RealMatrix lshBasis;
    private final int r;

    public FingerSearcher(HnswGraph hnswGraph, RandomAccessVectorValues<T> values, VectorEncoding encoding) {
        this.hnswGraph = hnswGraph;
        this.values = values;
        this.vectorEncoding = encoding;
        this.r = values.dimension() < 768 ? 64 : 128;

        // Compute the training data and the LSH basis.
        double[][] trainingData = new double[0][];
        try {
            trainingData = computeTrainingData();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        this.lshBasis = computeLshBasis(trainingData);
    }

    private double[][] computeTrainingData() throws IOException {
        int numVectors = values.size();
        int dimension = values.dimension();
        double[][] trainingData = new double[numVectors][];

        for (int i = 0; i < numVectors; i++) {
            T vector = values.vectorValue(i);
            double[] doubleVector;

            if (vectorEncoding == VectorEncoding.BYTE) {
                byte[] byteArray = (byte[]) vector;
                doubleVector = new double[byteArray.length];
                for (int j = 0; j < byteArray.length; j++) {
                    doubleVector[j] = byteArray[j];
                }
            } else if (vectorEncoding == VectorEncoding.FLOAT32) {
                float[] floatArray = (float[]) vector;
                doubleVector = new double[floatArray.length];
                for (int j = 0; j < floatArray.length; j++) {
                    doubleVector[j] = floatArray[j];
                }
            } else {
                throw new IllegalArgumentException("Unsupported vector encoding: " + vectorEncoding);
            }

            trainingData[i] = doubleVector;
        }

        return trainingData;
    }

    private RealMatrix computeLshBasis(double[][] trainingData) {
        // Compute the mean of the training data.
        VectorialMean mean = new VectorialMean(trainingData[0].length);
        for (double[] data : trainingData) {
            mean.increment(data);
        }
        double[] meanVector = mean.getResult();

        // Subtract the mean from the training data.
        for (double[] data : trainingData) {
            for (int i = 0; i < data.length; i++) {
                data[i] -= meanVector[i];
            }
        }

        // Compute the covariance matrix of the training data.
        PearsonsCorrelation pc = new PearsonsCorrelation(trainingData);
        RealMatrix covarianceMatrix = pc.getCorrelationMatrix();

        // Compute the eigen decomposition of the covariance matrix.
        EigenDecomposition eig = new EigenDecomposition(covarianceMatrix);

        // The LSH basis is given by the eigenvectors corresponding to the r largest eigenvalues.
        return eig.getV().getSubMatrix(0, r-1, 0, values.dimension()-1).transpose();
    }

    private float normSq(RealVector v) {
        return (float) v.dotProduct(v);
    }

    public Function<T, Float> distanceFunction(T query) {
        // Project the query vector into the LSH space
        RealVector queryVector;
        if (vectorEncoding == VectorEncoding.BYTE) {
            byte[] byteArray = (byte[]) query;
            queryVector = new ArrayRealVector(byteArray.length);
            for (int j = 0; j < byteArray.length; j++) {
                queryVector.setEntry(j, byteArray[j]);
            }
        } else if (vectorEncoding == VectorEncoding.FLOAT32) {
            float[] floatArray = (float[]) query;
            queryVector = new ArrayRealVector(floatArray.length);
            for (int j = 0; j < floatArray.length; j++) {
                queryVector.setEntry(j, floatArray[j]);
            }
        } else {
            throw new IllegalArgumentException("Unsupported vector encoding: " + vectorEncoding);
        }

        RealVector queryProjection = lshBasis.operate(queryVector);
        float queryProjectionLengthSquared = normSq(queryProjection);

        // Function that computes the distance between the query vector and other vectors in the LHS space
        return otherVector -> {
            RealVector otherVectorInLSHSpace;
            if (vectorEncoding == VectorEncoding.BYTE) {
                byte[] byteArray = (byte[]) otherVector;
                otherVectorInLSHSpace = new ArrayRealVector(byteArray.length);
                for (int j = 0; j < byteArray.length; j++) {
                    otherVectorInLSHSpace.setEntry(j, byteArray[j]);
                }
            } else {
                assert vectorEncoding == VectorEncoding.FLOAT32;
                float[] floatArray = (float[]) otherVector;
                otherVectorInLSHSpace = new ArrayRealVector(floatArray.length);
                for (int j = 0; j < floatArray.length; j++) {
                    otherVectorInLSHSpace.setEntry(j, floatArray[j]);
                }
            }

            RealVector otherProjection = lshBasis.operate(otherVectorInLSHSpace);
            float opSquared = normSq(otherProjection);

            // Compute the cosine similarity in the LSH space
            return queryProjectionLengthSquared + opSquared - 2 * (float) queryProjection.dotProduct(otherProjection);
        };
    }
}
