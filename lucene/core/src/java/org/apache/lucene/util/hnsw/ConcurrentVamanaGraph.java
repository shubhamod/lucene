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

import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.similarities.Similarity;
import org.apache.lucene.util.hnsw.ConcurrentNeighborSet.NeighborSimilarity;

import java.io.IOException;
import java.util.PrimitiveIterator;
import java.util.stream.IntStream;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;

/**
 * A single-level graph for implementing the Vamana algorithm.
 */
public final class ConcurrentVamanaGraph extends HnswGraph {
  private final int entryNode; // mediod of the graph

  private final ConcurrentNeighborSet[] graphNeighbors;

  ConcurrentVamanaGraph(HnswGraph hnsw, int M, int entryPoint, NeighborSimilarity similarityFunction) {
    this.entryNode = entryPoint;
    this.graphNeighbors = new ConcurrentNeighborSet[hnsw.size()];
    for (int i = 0; i < hnsw.size(); i++) {
      if (hnsw instanceof ConcurrentOnHeapHnswGraph) {
        graphNeighbors[i] = ((ConcurrentOnHeapHnswGraph) hnsw).getNeighbors(0, i).copy();
      } else {
        if (!(hnsw instanceof OnHeapHnswGraph)) {
          throw new IllegalArgumentException("Only OnHeapHnswGraph and ConcurrentOnHeapHnswGraph are supported");
        }
        var hhnsw = (OnHeapHnswGraph) hnsw;
        graphNeighbors[i] = new ConcurrentNeighborSet(i, hhnsw.getNeighbors(0, i), M, similarityFunction);
      }
    }
  }

  @Override
  public int size() {
    return graphNeighbors.length;
  }

  @Override
  public void seek(int level, int target) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public int nextNeighbor() throws IOException {
    throw new UnsupportedOperationException();
  }

  /**
   * @return the current number of levels in the graph where nodes have been added and we have a
   *     valid entry point.
   */
  @Override
  public int numLevels() {
    return 1;
  }

  /**
   * Returns the graph's current entry node on the top level shown as ordinals of the nodes on 0th
   * level
   *
   * @return the graph's current entry node on the top level
   */
  @Override
  public int entryNode() {
    return entryNode;
  }

  @Override
  public NodesIterator getNodesOnLevel(int level) {
    if (level != 0) {
      throw new IllegalArgumentException("Only level 0 is present in Vamana graphs");
    }
    PrimitiveIterator.OfInt it = IntStream.range(0, size()).iterator();
    return new NodesIterator(size()) {
      @Override
      public int consume(int[] dest) {
        throw new UnsupportedOperationException();
      }

      @Override
      public int nextInt() {
        return it.nextInt();
      }

      @Override
      public boolean hasNext() {
        return it.hasNext();
      }
    };
  }

  @Override
  public String toString() {
    return "VamanaGraph(size=" + size() + ", entryNode=" + entryNode;
  }

  /**
   * Returns a view of the graph that is safe to use concurrently with updates performed on the
   * underlying graph.
   *
   * <p>Multiple Views may be searched concurrently.
   */
  public HnswGraph getView() {
    return new ConcurrentVamanaGraphView();
  }

  public ConcurrentNeighborSet getNeighbors(int node) {
    return graphNeighbors[node];
  }

  private class ConcurrentVamanaGraphView extends HnswGraph {
    private PrimitiveIterator.OfInt remainingNeighbors;

    @Override
    public int size() {
      return ConcurrentVamanaGraph.this.size();
    }

    @Override
    public int numLevels() {
      return ConcurrentVamanaGraph.this.numLevels();
    }

    @Override
    public int entryNode() {
      return ConcurrentVamanaGraph.this.entryNode();
    }

    @Override
    public NodesIterator getNodesOnLevel(int level) {
      return ConcurrentVamanaGraph.this.getNodesOnLevel(level);
    }

    @Override
    public void seek(int level, int targetNode) {
      remainingNeighbors = graphNeighbors[targetNode].nodeIterator();
    }

    @Override
    public int nextNeighbor() {
      if (remainingNeighbors.hasNext()) {
        return remainingNeighbors.nextInt();
      }
      return NO_MORE_DOCS;
    }

    @Override
    public String toString() {
      return "ConcurrentVamanaGraphView(size=" + size() + ", entryPoint=" + entryNode;
    }
  }
}
