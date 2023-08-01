package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.VectorUtil;
import org.apache.lucene.util.hnsw.math.distribution.NormalDistribution;
import org.apache.lucene.util.hnsw.math.linear.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.stream.IntStream;

import static java.lang.Math.sqrt;
import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.apache.lucene.util.VectorUtil.dotProduct;
import static org.apache.lucene.util.hnsw.VectorMath.cosine;
import static org.apache.lucene.util.hnsw.VectorMath.mapMultiply;
import static org.apache.lucene.util.hnsw.VectorMath.norm;
import static org.apache.lucene.util.hnsw.VectorMath.subtract;
import static org.apache.lucene.util.hnsw.VectorMath.toFloatArray;
import static org.apache.lucene.util.hnsw.VectorMath.toFloatMatrix;

public class FingerMetadata<T> {
    private final RandomAccessVectorValues<T> vectors;
    private final VectorEncoding vectorEncoding;
    private final HnswGraph hnsw;
    private final int lshDimensions;
    private final Map<Integer, CachedResidual>[] cachedResiduals;
    private final VectorSimilarityFunction similarityFunction;
    private final LshBasis lsh;
    private final LinearTransform transform;
    private final float[] cNormSquared;
    private final float[][] cBasis; // projection of each vector onto the LSH basis
    private final float piOverDimensions;

    public FingerMetadata(HnswGraph hnswGraph, RandomAccessVectorValues<T> vectors, VectorEncoding encoding, VectorSimilarityFunction similarityFunction, int lshDimensions) {
        if (similarityFunction != VectorSimilarityFunction.DOT_PRODUCT) {
            // we've only implemented the FINGER algorithm for dot product so far
            throw new IllegalArgumentException("Unsupported similarity function: " + similarityFunction);
        }
        if (!(hnswGraph instanceof OnHeapHnswGraph || hnswGraph instanceof ConcurrentOnHeapHnswGraph.ConcurrentHnswGraphView)) {
            throw new IllegalArgumentException("We can only compute FINGER metadata for on-heap hnsw graphs");
        }

        this.hnsw = hnswGraph;
        this.vectorEncoding = encoding;
        this.lshDimensions = lshDimensions;
        this.piOverDimensions = (float) (Math.PI / lshDimensions);
        this.similarityFunction = similarityFunction;
        // Compute the training data and the LSH basis.
        try {
            this.vectors = vectors.copy();
//            this.lsh = LshBasis.createRandom(vectors.dimension(), lshDimensions);
            this.lsh = LshBasis.computeFromResiduals(new TrainingVectorIterator(), vectors.dimension(), lshDimensions);
            this.cNormSquared = cacheCNormSquared();
            this.cBasis = cacheCBasis();
            this.transform = computeTransform();
            this.cachedResiduals = cacheResiduals();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private float[][] cacheCBasis() throws IOException {
        float[][] cBasis = new float[vectors.size()][];
        for (int i = 0; i < vectors.size(); i++) {
            float[] c = (float[]) vectors.vectorValue(i);
            cBasis[i] = lsh.project(c);
        }
        return cBasis;
    }

    private float[] cacheCNormSquared() throws IOException {
        float[] cNormSquared = new float[vectors.size()];
        for (int i = 0; i < vectors.size(); i++) {
            float[] v = (float[]) vectors.vectorValue(i);
            cNormSquared[i] = VectorUtil.dotProduct(v, v);
        }
        return cNormSquared;
    }

    private Map<Integer, CachedResidual>[] cacheResiduals() throws IOException {
        var vectorsCopy = vectors.copy();
        Map<Integer, CachedResidual>[] map = new Map[hnsw.size()];
        for (int level = 0; level < hnsw.numLevels(); level++)
        {
            HnswGraph.NodesIterator it = hnsw.getNodesOnLevel(level);
            while (it.hasNext()) {
                int node = it.nextInt();
                if (map[node] == null) {
                    map[node] = new HashMap<>();
                }
                T c = vectors.vectorValue(node);
                hnsw.seek(level, node);
                int neighbor;
                while ((neighbor = hnsw.nextNeighbor()) != NO_MORE_DOCS) {
                    // Compute the residual vector.
                    T d = vectorsCopy.vectorValue(neighbor);
                    map[node].put(neighbor, new CachedResidual(node, (float[]) c, (float[]) d));
                }
            }
        }
        return map;
    }

    private LinearTransform computeTransform() throws IOException {
        List<Double> estimatedAngles = new ArrayList<>();
        List<Double> actualAngles = new ArrayList<>();
        for (int i = 0; i < hnsw.size(); i++) {
            hnsw.seek(0, i);
            int queryNode = hnsw.nextNeighbor(); // we'll compute correlation around the first neighbor
            int neighbor = hnsw.nextNeighbor();
            if (neighbor == NO_MORE_DOCS) {
                continue;
            }
            var ea = estimateAngleForTraining(i, queryNode, neighbor);
            if (Double.isFinite(ea.actual) && Double.isFinite(ea.estimated)) {
                // checking for NaN is a workaround for SIFT dataset containing duplicate vectors
                // -- normally higher level is responsible for dedup but I want to run a test against the raw Hnsw
                // classes
                estimatedAngles.add(ea.estimated);
                actualAngles.add(ea.actual);
            }
        }

        // Calculate mean and standard deviation for estimated and actual similarities
        double estimatedMean = estimatedAngles.stream().mapToDouble(val -> val).average().orElse(0.0);
        double actualMean = actualAngles.stream().mapToDouble(val -> val).average().orElse(0.0);
        double estimatedStdDev = sqrt(estimatedAngles.stream().mapToDouble(val -> (val - estimatedMean) * (val - estimatedMean)).average().orElse(0.0));
        double actualStdDev = sqrt(actualAngles.stream().mapToDouble(val -> (val - actualMean) * (val - actualMean)).average().orElse(0.0));
        double epsilon = IntStream.range(0, estimatedAngles.size()).mapToDouble(i -> {
            return (estimatedAngles.get(i) - estimatedMean) * (actualStdDev / estimatedStdDev) + actualMean - actualAngles.get(i);
        }).sum() / estimatedAngles.size();
        return new LinearTransform(estimatedMean, actualMean, estimatedStdDev, actualStdDev, epsilon);
    }

    static class LinearTransform {
        public final double fromMean, toMeanEspilon, stdDevRatio;

        LinearTransform(double fromMean, double toMean, double fromStdDev, double toStdDev, double epsilon) {
            this.fromMean = fromMean;
            this.stdDevRatio = toStdDev / fromStdDev;
            this.toMeanEspilon = toMean + epsilon;
        }

        public double apply(double a) {
            return (a - fromMean) * (stdDevRatio) + toMeanEspilon;
        }
    }

    static class LshBasis {
        public final float[][] basis;

        public LshBasis(float[][] basis) {
            this.basis = basis;
        }

        public float[] project(float[] v) {
            int dim = basis.length;
            float[] projection = new float[dim];
            for (int i = 0; i < dim; i++) {
                projection[i] = dotProduct(basis[i], v);
            }
            return projection;
        }

        public static LshBasis createRandom(int originalDimensions, int basisDimensions) {
            double[][] randomData = new double[originalDimensions][basisDimensions];
            NormalDistribution nd = new NormalDistribution();
            for (int j = 0; j < originalDimensions; j++) {
                for (int k = 0; k < basisDimensions; k++) {
                    randomData[j][k] = nd.sample();
                }
            }
            // Compute the QR decomposition of the random matrix.
            QRDecomposition qr = new QRDecomposition(new Array2DRowRealMatrix(randomData));
            // The Q matrix is a 50x50 random orthogonal matrix. We keep only the first 8 columns.
            RealMatrix Q = qr.getQ().getSubMatrix(0, originalDimensions - 1, 0, basisDimensions - 1);
            return new LshBasis(toFloatMatrix(Q.transpose()));
        }

        static <T> LshBasis computeFromResiduals(Iterator<float[]> data, int dataDimensions, int lshDimensions) {
            double[][] covarianceMatrix = VectorMath.incrementalCovariance(data, dataDimensions);

            // Compute the eigen decomposition of the covariance matrix.
            RealMatrix covarianceMatrixRM = MatrixUtils.createRealMatrix(covarianceMatrix);
            EigenDecomposition eig = new EigenDecomposition(covarianceMatrixRM);

            // The LSH basis is given by the eigenvectors corresponding to the r largest eigenvalues.
            // Get the eigenvalues and the matrix of eigenvectors.
            double[] eigenvalues = eig.getRealEigenvalues();
            RealMatrix eigenvectors = eig.getV();

            // Create a stream of indices [0, 1, 2, ..., n-1], sort them by corresponding eigenvalue in descending order,
            // and select the top lshDimensions indices.
            int[] topIndices = IntStream.range(0, eigenvalues.length)
                .boxed()
                .sorted(Comparator.comparingDouble(i -> -eigenvalues[i]))
                .mapToInt(Integer::intValue)
                .limit(lshDimensions)
                .toArray();

            // Extract the corresponding eigenvectors.
            float[][] basis = new float[lshDimensions][];
            for (int i = 0; i < lshDimensions; i++) {
                RealVector eigenvector = eigenvectors.getColumnVector(topIndices[i]);
                basis[i] = toFloatArray(eigenvector);
            }

            return new LshBasis(basis);
        }
    }

    public HnswSearcher.SimilarityProvider similarityProviderFor(T query) {
        return new FingerSimilarityProvider(query);
    }

    static class EstimatedAngle {
        public final double estimated;
        public final double actual;

        EstimatedAngle(double estimated, double actual) {
            this.estimated = estimated;
            this.actual = actual;
        }
    }

    private EstimatedAngle estimateAngleForTraining(int cNode, int d1Node, int d2Node) throws IOException {
        var v1 = vectors.copy();
        var v2 = vectors.copy();
        var cNS = cNormSquared[cNode];
        var c = (float[]) vectors.vectorValue(cNode);
        var q = (float[]) v1.vectorValue(d1Node);
        var d = (float[]) v2.vectorValue(d2Node);

        var qProj = mapMultiply(c, dotProduct(q, c) / cNS);
        var qRes = subtract(q, qProj);
        var qResB = lsh.project(qRes);

        var dProj = mapMultiply(c, dotProduct(d, c) / cNS);
        var dRes = subtract(d, dProj);
        var dResB = lsh.project(dRes);

        // TODO can we used the cached bits for this?
        int diffSigns = 0;
        for (int i = 0; i < lshDimensions; i++) {
            if ((qResB[i] > 0) != (dResB[i] > 0)) {
                diffSigns++;
            }
        }
        return new EstimatedAngle(Math.cos(diffSigns * Math.PI / lshDimensions), cosine(qRes, dRes));
    }

    private class TrainingVectorIterator implements Iterator<float[]> {
        int level = 0;
        HnswGraph.NodesIterator nodesIterator;
        int currentNode;
        T c;
        int neighbor;

        private TrainingVectorIterator() throws IOException {
            nodesIterator = hnsw.getNodesOnLevel(level);
            currentNode = nodesIterator.hasNext() ? nodesIterator.nextInt() : NO_MORE_DOCS;
            c = currentNode != NO_MORE_DOCS ? vectors.vectorValue(currentNode) : null;
            hnsw.seek(level, currentNode);
            neighbor = hnsw.nextNeighbor();
        }

        @Override
        public boolean hasNext() {
            return currentNode != NO_MORE_DOCS && neighbor != NO_MORE_DOCS;
        }

        @Override
        public float[] next() {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }

            try {
                return nextInternal();
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        private float[] nextInternal() throws IOException {
            // Compute the residual vector
            float[] d = (float[]) vectors.vectorValue(neighbor);
            float scale = dotProduct(d, (float[]) c);
            float[] dProj = mapMultiply((float[]) c, scale);
            float[] dRes = subtract(d, dProj);

            // Move to the next neighbor or node or level if necessary
            if ((neighbor = hnsw.nextNeighbor()) == NO_MORE_DOCS) {
                if (nodesIterator.hasNext()) {
                    currentNode = nodesIterator.nextInt();
                    c = vectors.vectorValue(currentNode);
                    hnsw.seek(level, currentNode);
                    neighbor = hnsw.nextNeighbor();
                } else if (++level < hnsw.numLevels()) {
                    nodesIterator = hnsw.getNodesOnLevel(level);
                    currentNode = nodesIterator.hasNext() ? nodesIterator.nextInt() : NO_MORE_DOCS;
                    if (currentNode != NO_MORE_DOCS) {
                        c = vectors.vectorValue(currentNode);
                        hnsw.seek(level, currentNode);
                        neighbor = hnsw.nextNeighbor();
                    }
                } else {
                    currentNode = NO_MORE_DOCS;
                }
            }

            return dRes;
        }
    }

    private class CachedResidual {
        private final long dResBits;
        private final float dResNorm;
        private final float dDotC;

        public CachedResidual(int cNode, float[] c, float[] d) {
            float cNS = cNormSquared[cNode];
            dDotC = dotProduct(d, c);
            float[] dProj = mapMultiply(c, dDotC / cNS);
            float[] dRes = subtract(d, dProj);
            dResBits = toBits(lsh.project(dRes));
            dResNorm = norm(dRes);
        }

        public static long toBits(float[] v) {
            assert v.length <= 64;
            long bits = 0;
            for (int i = 0; i < v.length; i++) {
                if (v[i] > 0) {
                    bits |= (1L << i);
                }
            }
            return bits;
        }
    }

    private class FingerSimilarityProvider implements HnswSearcher.SimilarityProvider {
        private final T q;
        private final float qNS;
        private final float[] Bq;

        public FingerSimilarityProvider(T query) {
            this.q = query;
            this.qNS = VectorUtil.dotProduct((float[]) q, (float[]) q);
            Bq = lsh.project((float[]) q);
        }

        @Override
        public float exactSimilarityTo(int node) throws IOException {
            if (vectorEncoding == VectorEncoding.BYTE) {
                return similarityFunction.compare((byte[]) q, (byte[]) vectors.vectorValue(node));
            } else {
                return similarityFunction.compare((float[]) q, (float[]) vectors.vectorValue(node));
            }
        }

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
        @Override
        public SimilarityFunction approximateSimilarityNear(int cNode, float qDotC) {
            float cNS = cNormSquared[cNode];

            // "We know that norm(q_proj)^2 = t^2 * norm(c)^2 [where t = (q.c / c.c)] ...
            //  Since norm(q)^2 = norm(q_proj)^2 + norm(q_res)^2,
            //  norm(q_res)^2 = norm(q)^2 - norm(q_proj)^2 = norm(q)^2 - t^2 * norm(c)^2"
            float t = qDotC / cNS;
            float qResNorm = (float) sqrt(qNS - t * t * cNS);

            // "To get Bq_res, recall q_res = q - q_proj, so
            //  Bq_res = Bq - Bq_proj = Bq - tBc [where t = (q.c / c.c)]"
            float[] Bc = cBasis[cNode];
            float[] Bq_res = subtract(Bq, mapMultiply(Bc, t));
            long qResBits = CachedResidual.toBits(Bq_res);

            return neighbor -> {
                CachedResidual cr = cachedResiduals[cNode].get(neighbor);
                float projectedTerm = t * cr.dDotC;

                // Estimate angle using Lemma 1
                int diffSigns = Long.bitCount(qResBits ^ cr.dResBits);
                float estimatedAngle = diffSigns * piOverDimensions;
                float adjustedCosine = (float) transform.apply(Math.cos(estimatedAngle));

                float residualTerm = qResNorm * cr.dResNorm * adjustedCosine;
                float approximateDotProduct = projectedTerm + residualTerm;

                return (1 + approximateDotProduct) / 2;
            };
        }
    }
}
