package org.apache.lucene.util.hnsw;

import org.apache.lucene.index.VectorEncoding;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

public class FingerSearcher<T> {

    private final VectorEncoding vectorEncoding;

    private final float[][] lshBases;

    public FingerSearcher(HnswGraph graph, RandomAccessVectorValues<T> vectors, VectorEncoding encoding) throws IOException {

        this.vectorEncoding = encoding;

        float[][] residualVectors = computeResidualVectors(graph, vectors);

        lshBases = learnLSHBases(residualVectors);
    }

    private float[][] computeResidualVectors(HnswGraph graph, RandomAccessVectorValues<T> vectors) throws IOException {

        List<int[]> edges = getAllEdges(graph);

        float[][] residuals = new float[edges.size()][];
        for(int i = 0; i < edges.size(); i++) {
            int[] edge = edges.get(i);
            residuals[i] = getResidual(edge[0], edge[1], vectors);
        }

        return residuals;

    }

    private float[] getResidual(int node1, int node2, RandomAccessVectorValues<T> vectors) throws IOException {
        float[] vec1, vec2, proj, residual;

        if(vectorEncoding == VectorEncoding.FLOAT32) {
            float[] node1Vec = (float[]) vectors.vectorValue(node1);
            float[] node2Vec = (float[]) vectors.vectorValue(node2);
            vec1 = node1Vec;
            vec2 = node2Vec;

        } else {
            byte[] node1Vec = (byte[]) vectors.vectorValue(node1);
            byte[] node2Vec = (byte[]) vectors.vectorValue(node2);
            vec1 = byteArrayToFloatArray(node1Vec);
            vec2 = byteArrayToFloatArray(node2Vec);
        }

        proj = project(vec2, vec1);
        residual = subtract(vec2, proj);

        return residual;

    }

    private List<int[]> getAllEdges(HnswGraph graph) throws IOException {

        List<int[]> edges = new ArrayList<>();

        for(int level = 0; level < graph.numLevels(); level++) {
            HnswGraph.NodesIterator nodes = graph.getNodesOnLevel(level);
            while(nodes.hasNext()) {
                int node = nodes.next();
                graph.seek(level, node);

                int neighbor;
                while((neighbor = graph.nextNeighbor()) != NO_MORE_DOCS) {
                    edges.add(new int[]{node, neighbor});
                }
            }
        }

        return edges;

    }

    private float[] project(float[] vec, float[] onto) {

        // Compute projection of vec onto onto
        float[] proj = new float[vec.length];

        double ontoNorm = Math.sqrt(dotProduct(onto, onto));
        for(int i = 0; i < vec.length; i++) {
            proj[i] = (float) ((dotProduct(vec, onto) / ontoNorm) * onto[i]);
        }

        return proj;

    }

    private float dotProduct(float[] vec1, float[] vec2) {
        float product = 0;
        for(int i = 0; i < vec1.length; i++) {
            product += vec1[i] * vec2[i];
        }
        return product;
    }

    private float[] subtract(float[] vec1, float[] vec2) {

        // Compute vec1 - vec2
        float[] result = new float[vec1.length];
        for(int i = 0; i < vec1.length; i++) {
            result[i] = vec1[i] - vec2[i];
        }

        return result;

    }

    private float[] byteArrayToFloatArray(byte[] input) {

        float[] output = new float[input.length];

        for(int i = 0; i < input.length; i++) {
            output[i] = input[i]; // Bytes to floats
        }

        return output;

    }

    private float[][] learnLSHBases(float[][] residualVectors) {

        return pca(residualVectors, 64);
    }

    private float[][] pca(float[][] data, int numBases) {
        // Subtract mean
        float[] mean = computeMean(data);
        float[][] centered = centerData(data, mean);

        // Compute covariance matrix
        float[][] covariance = computeCovariance(centered);

        // Compute eigenvectors of covariance matrix
        EigenvalueDecomposition evd = new EigenvalueDecomposition(covariance);
        float[][] eigenvectors = evd.getEigenvectors();

        // Take top k eigenvectors as bases
        float[][] bases = new float[numBases][];
        System.arraycopy(eigenvectors, 0, bases, 0, numBases);

        return bases;
    }

    private float[] computeMean(float[][] data) {
        float[] mean = new float[data[0].length];
        for(float[] row : data) {
            for(int j = 0; j < mean.length; j++) {
                mean[j] += row[j];
            }
        }
        for(int i = 0; i < mean.length; i++) {
            mean[i] /= data.length;
        }
        return mean;
    }

    private float[][] centerData(float[][] data, float[] mean) {
        float[][] centered = new float[data.length][];
        for(int i = 0; i < data.length; i++) {
            centered[i] = subtract(data[i], mean);
        }
        return centered;
    }

    private float[][] computeCovariance(float[][] centered) {
        float[][] covariance = new float[centered[0].length][centered[0].length];
        for(float[] row : centered) {
            float[] xt = transpose(row);
            covariance = sum(covariance, multiply(row, xt));
        }
        covariance = scale(covariance, 1.0f/centered.length);
        return covariance;
    }

    private float[] transpose(float[] vec) {
        float[] result = new float[vec.length];
        System.arraycopy(vec, 0, result, 0, vec.length);
        return result;
    }

    private float[][] sum(float[][] mat1, float[][] mat2) {
        float[][] result = new float[mat1.length][mat1[0].length];
        for(int i = 0; i < result.length; i++) {
            for(int j = 0; j < result[0].length; j++) {
                result[i][j] = mat1[i][j] + mat2[i][j];
            }
        }
        return result;
    }

    private float[][] multiply(float[] vec, float[] vecT) {
        float[][] result = new float[vec.length][vecT.length];
        for(int i = 0; i < vec.length; i++) {
            for(int j = 0; j < vecT.length; j++) {
                result[i][j] = vec[i] * vecT[j];
            }
        }
        return result;
    }

    private float[][] scale(float[][] mat, float scale) {
        float[][] result = new float[mat.length][mat[0].length];
        for(int i = 0; i < mat.length; i++) {
            for(int j = 0; j < mat[0].length; j++) {
                result[i][j] = mat[i][j] * scale;
            }
        }
        return result;
    }

    public float[][] getLSHBases() {
        return lshBases;
    }
}
