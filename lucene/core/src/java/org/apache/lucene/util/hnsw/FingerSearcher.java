package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.hnsw.math.linear.*;
import org.apache.lucene.util.hnsw.math.stat.correlation.PearsonsCorrelation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

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
        List<RealVector> trainingData = null;
        try {
            trainingData = computeTrainingData();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        this.lshBasis = computeLshBasis(trainingData);
    }

    private List<RealVector> computeTrainingData() throws IOException {
        int numNodes = hnswGraph.size();
        List<RealVector> trainingData = new ArrayList<>(numNodes);
        RealMatrix valuesMatrix = new Array2DRowRealMatrix(values.size(), values.dimension());

        for (int i = 0; i < values.size(); i++) {
            T vector = values.vectorValue(i);

            // Convert vector to double array based on vectorEncoding.
            double[] doubleVector;
            if (vectorEncoding == VectorEncoding.BYTE) {
                byte[] byteVector = (byte[]) vector;
                doubleVector = new double[byteVector.length];
                for (int j = 0; j < byteVector.length; j++) {
                    doubleVector[j] = byteVector[j];
                }
            } else if (vectorEncoding == VectorEncoding.FLOAT32) {
                float[] floatVector = (float[]) vector;
                doubleVector = new double[floatVector.length];
                for (int j = 0; j < floatVector.length; j++) {
                    doubleVector[j] = floatVector[j];
                }
            } else {
                throw new IllegalStateException("Unsupported vector encoding: " + vectorEncoding);
            }

            valuesMatrix.setRowVector(i, new ArrayRealVector(doubleVector));
        }

        for (int node = 0; node < numNodes; node++) {
            RealVector c = valuesMatrix.getRowVector(node);
            hnswGraph.seek(0, node);
            int neighbor;
            while ((neighbor = hnswGraph.nextNeighbor()) != NO_MORE_DOCS) {
                RealVector d = valuesMatrix.getRowVector(neighbor);

                // Compute the residual vector.
                double[] projection = c.toArray();
                double scale = d.dotProduct(c) / c.dotProduct(c);
                for (int i = 0; i < projection.length; i++) {
                    projection[i] *= scale;
                }
                RealVector d_proj = new ArrayRealVector(projection);
                RealVector d_res = d.subtract(d_proj);

                trainingData.add(d_res);
            }
        }
        return trainingData;
    }

    private RealMatrix computeLshBasis(List<RealVector> trainingData) {
        // Compute the mean of the training data
        double[] meanVector = new double[values.dimension()];
        for (RealVector data : trainingData) {
            for (int i = 0; i < data.getDimension(); i++) {
                meanVector[i] += data.getEntry(i);
            }
        }
        for (int i = 0; i < meanVector.length; i++) {
            meanVector[i] /= trainingData.size();
        }

        // Subtract the mean from the training data.
        for (RealVector data : trainingData) {
            for (int i = 0; i < data.getDimension(); i++) {
                data.setEntry(i, data.getEntry(i) - meanVector[i]);
            }
        }

        // Convert the training data back to a 2D array for PearsonsCorrelation.
        double[][] centeredData = new double[trainingData.size()][];
        for (int i = 0; i < trainingData.size(); i++) {
            centeredData[i] = trainingData.get(i).toArray();
        }

        // Compute the covariance matrix of the training data.
        PearsonsCorrelation pc = new PearsonsCorrelation(centeredData);
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
