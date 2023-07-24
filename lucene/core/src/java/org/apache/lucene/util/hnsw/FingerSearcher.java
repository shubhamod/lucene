package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.hnsw.math.linear.*;
import org.apache.lucene.util.hnsw.math.stat.correlation.PearsonsCorrelation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class FingerSearcher<T> {
    private final RandomAccessVectorValues<T> values;
    private final VectorEncoding vectorEncoding;
    private final HnswGraph hnswGraph;
    private final RealMatrix lshBasis;
    private final int lshDimensions;
    private final Map<Integer, RealVector>[] neighorResiduals; // d_res for each node -> neighbor

    public FingerSearcher(HnswGraph hnswGraph, RandomAccessVectorValues<T> values, VectorEncoding encoding) {
        this.hnswGraph = hnswGraph;
        this.values = values;
        this.vectorEncoding = encoding;
        this.lshDimensions = values.dimension() <= 768 ? 64 : 128;
        this.neighorResiduals = new Map[hnswGraph.size()];
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
            var residuals = new HashMap<Integer, RealVector>();
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
                residuals.put(neighbor, d_res);
            }

            neighorResiduals[node] = residuals;
        }
        return trainingData;
    }

    private RealMatrix valuesToMatrix() throws IOException {
        RealMatrix valuesMatrix = new Array2DRowRealMatrix(values.size(), values.dimension());
        for (int i = 0; i < values.size(); i++) {
            RealVector v = toRealVector(values.vectorValue(i));
            valuesMatrix.setRowVector(i, v);
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
        // The first FINGER insight is that we can compute the distance D = q.d
        // as qp.dp + qr.dr, where qp and dp are the projections of the query vector and the data
        // (neighbor) vector onto the node vector c, and qr and dr are the residuals of that projection.
        // (I am using x.y to indicate dot product of x and y.)
        //
        // This is valuable because we can express the first term as easily-cached and easily-computed
        // operations, compared to calculating the full n-dimensional dot product of q and d.
        // qp.dp = (qp.c / c.c) * c . (dp.c / c.c) * c
        //       = (qp.c * dp.c) / (c.c)^2
        // qp.c is the projection of q onto c, which we compute once at the start of the query.
        // dp.c is the projection of d onto c, which we compute once for each neighbor and cache.
        // c.c is the norm of c, which we compute once for each node and cache.
        //
        // The second term is the dot product of the residuals.  There is no way to avoid computing
        // this, BUT the second FINGER insight is that we can approximate this term using LSH with
        // very little loss of precision.
        RealVector queryVector = toRealVector(query);

        float projected =
        // Project the query vector into the LSH space
        RealVector queryProjection = lshBasis.operate(queryVector);
        float qpSquared = normSq(queryProjection);

        // Function that computes the distance between the query vector and other vectors in the LHS space
        return ordinal -> {
            RealVector v = null;
            try {
                v = toRealVector(values.vectorValue(ordinal));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            // Get d_res for this neighbor
            RealVector dRes = neighorResiduals[ordinal].get(ordinal);

            // Compute q_res
            RealVector qRes = queryVector.subtract(queryProjection);

            // Compute LSH hash for d_res and q_res
            byte[] dLSH = lshBasis.operate(dRes).map(v -> v > 0 ? 1 : 0);
            byte[] qLSH = lshBasis.operate(qRes).map(v -> v > 0 ? 1 : 0);

            // Estimate angle between q_res and d_res
            float estAngle = lshApproxAngle(qLSH, dLSH);

            // Compute approximate q_res^T d_res term
            float qdTerm = normSq(qRes) * normSq(dRes) * estAngle;

            // Combine with other terms
            return qProjDiffTerm + qdTerm;
        };
    }

    private RealVector toRealVector(T query) {
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
        return queryVector;
    }
}
