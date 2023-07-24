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
    private final int lshDimensions;

    public FingerSearcher(HnswGraph hnswGraph, RandomAccessVectorValues<T> values, VectorEncoding encoding) {
        this.hnswGraph = hnswGraph;
        this.values = values;
        this.vectorEncoding = encoding;
        this.lshDimensions = values.dimension() <= 768 ? 64 : 128;

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
        List<RealVector> trainingData = new ArrayList<>(hnswGraph.size());
        RealMatrix valuesMatrix = valuesToMatrix();

        // for each node, create a training residual with each of its neighbors
        for (int node = 0; node < hnswGraph.size(); node++) {
            RealVector c = valuesMatrix.getRowVector(node);
            hnswGraph.seek(0, node);
            int neighbor;
            while ((neighbor = hnswGraph.nextNeighbor()) != NO_MORE_DOCS) {
                // Compute the residual vector.
                RealVector d = valuesMatrix.getRowVector(neighbor);
                double scale = d.dotProduct(c) / c.dotProduct(c);
                RealVector d_proj = c.mapMultiply(scale);
                RealVector d_res = d.subtract(d_proj);

                trainingData.add(d_res);
            }
        }
        return trainingData;
    }

    private RealMatrix valuesToMatrix() throws IOException {
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
        return valuesMatrix;
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
        return eig.getV().getSubMatrix(0, lshDimensions -1, 0, values.dimension()-1).transpose();
    }

    private float normSq(RealVector v) {
        return (float) v.dotProduct(v);
    }

    public Function<Integer, Float> distanceFunction(T query) {
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
        float qpSquared = normSq(queryProjection);

        // Function that computes the distance between the query vector and other vectors in the LHS space
        return ordinal -> {
            T genericVector = null;
            try {
                genericVector = values.vectorValue(ordinal);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            RealVector v;
            if (vectorEncoding == VectorEncoding.BYTE) {
                byte[] byteArray = (byte[]) genericVector;
                v = new ArrayRealVector(byteArray.length);
                for (int j = 0; j < byteArray.length; j++) {
                    v.setEntry(j, byteArray[j]);
                }
            } else {
                assert vectorEncoding == VectorEncoding.FLOAT32;
                float[] floatArray = (float[]) genericVector;
                v = new ArrayRealVector(floatArray.length);
                for (int j = 0; j < floatArray.length; j++) {
                    v.setEntry(j, floatArray[j]);
                }
            }

            RealVector vProjection = lshBasis.operate(v);
            float vpSquared = normSq(vProjection);

            // Compute the cosine similarity in the LSH space
            return qpSquared + vpSquared - 2 * (float) queryProjection.dotProduct(vProjection);
        };
    }
}
