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

import static java.lang.Math.log;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.function.Supplier;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.GrowableBitSet;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.ThreadInterruptedException;
import org.apache.lucene.util.hnsw.ConcurrentNeighborSet.NeighborSimilarity;
import org.apache.lucene.util.hnsw.ConcurrentOnHeapHnswGraph.NodeAtLevel;

/**
 * Builder for Concurrent HNSW graph. See {@link HnswGraph} for a high level overview, and the
 * comments to `addGraphNode` for details on the concurrent building approach.
 *
 * @param <T> the type of vector
 */
public class ConcurrentHnswGraphBuilder<T> {

  /** Default number of maximum connections per node */
  public static final int DEFAULT_MAX_CONN = 16;

  /**
   * Default number of the size of the queue maintained while searching during a graph construction.
   */
  public static final int DEFAULT_BEAM_WIDTH = 100;

  /** A name for the HNSW component for the info-stream */
  public static final String HNSW_COMPONENT = "HNSW";

  private final int beamWidth;
  private final double ml;
  private final ExplicitThreadLocal<NeighborArray> scratchNeighbors;

  private final VectorSimilarityFunction similarityFunction;
  private final VectorEncoding vectorEncoding;
  private final ExplicitThreadLocal<RandomAccessVectorValues<T>> vectors;
  private final ExplicitThreadLocal<HnswGraphSearcher<T>> graphSearcher;
  private final ExplicitThreadLocal<NeighborQueue> beamCandidates;

  final ConcurrentOnHeapHnswGraph hnsw;
  private final ConcurrentSkipListSet<NodeAtLevel> insertionsInProgress =
      new ConcurrentSkipListSet<>();

  private InfoStream infoStream = InfoStream.getDefault();

  // we need two sources of vectors in order to perform diversity check comparisons without
  // colliding
  private final ExplicitThreadLocal<RandomAccessVectorValues<T>> vectorsCopy;

  /** This is the "native" factory for ConcurrentHnswGraphBuilder. */
  public static <T> ConcurrentHnswGraphBuilder<T> create(
      RandomAccessVectorValues<T> vectors,
      VectorEncoding vectorEncoding,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth)
      throws IOException {
    return new ConcurrentHnswGraphBuilder<>(
        vectors, vectorEncoding, similarityFunction, M, beamWidth);
  }

  /**
   * Reads all the vectors from vector values, builds a graph connecting them by their dense
   * ordinals, using the given hyperparameter settings, and returns the resulting graph.
   *
   * @param vectorValues the vectors whose relations are represented by the graph - must provide a
   *     different view over those vectors than the one used to add via addGraphNode.
   * @param M – graph fanout parameter used to calculate the maximum number of connections a node
   *     can have – M on upper layers, and M * 2 on the lowest level.
   * @param beamWidth the size of the beam search to use when finding nearest neighbors.
   */
  public ConcurrentHnswGraphBuilder(
      RandomAccessVectorValues<T> vectorValues,
      VectorEncoding vectorEncoding,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth) {
    this.vectors = createThreadSafeVectors(vectorValues);
    this.vectorsCopy = createThreadSafeVectors(vectorValues);
    this.vectorEncoding = Objects.requireNonNull(vectorEncoding);
    this.similarityFunction = Objects.requireNonNull(similarityFunction);
    if (M <= 0) {
      throw new IllegalArgumentException("maxConn must be positive");
    }
    if (beamWidth <= 0) {
      throw new IllegalArgumentException("beamWidth must be positive");
    }
    this.beamWidth = beamWidth;
    // normalization factor for level generation; currently not configurable
    this.ml = M == 1 ? 1 : 1 / Math.log(1.0 * M);

    NeighborSimilarity similarity =
        new NeighborSimilarity() {
          @Override
          public float score(int node1, int node2) {
            try {
              return scoreBetween(
                  vectors.get().vectorValue(node1), vectorsCopy.get().vectorValue(node2));
            } catch (IOException e) {
              throw new UncheckedIOException(e);
            }
          }

          @Override
          public Function<Integer, Float> scoreProvider(int node1) {
            T v1;
            try {
              v1 = vectors.get().vectorValue(node1);
            } catch (IOException e) {
              throw new UncheckedIOException(e);
            }
            return node2 -> {
              try {
                return scoreBetween(v1, vectorsCopy.get().vectorValue(node2));
              } catch (IOException e) {
                throw new UncheckedIOException(e);
              }
            };
          }
        };
    this.hnsw = new ConcurrentOnHeapHnswGraph(M, similarity);

    this.graphSearcher =
        ExplicitThreadLocal.withInitial(
            () -> {
              return new HnswGraphSearcher<>(
                  vectorEncoding,
                  similarityFunction,
                  new NeighborQueue(beamWidth, true),
                  new GrowableBitSet(this.vectors.get().size()));
            });
    // in scratch we store candidates in reverse order: worse candidates are first
    this.scratchNeighbors =
        ExplicitThreadLocal.withInitial(() -> new NeighborArray(Math.max(beamWidth, M + 1), false));
    this.beamCandidates =
        ExplicitThreadLocal.withInitial(() -> new NeighborQueue(beamWidth, false));
  }

  private abstract static class ExplicitThreadLocal<U> {
    private final ConcurrentHashMap<Long, U> map = new ConcurrentHashMap<>();

    public U get() {
      return map.computeIfAbsent(Thread.currentThread().getId(), k -> initialValue());
    }

    protected abstract U initialValue();

    public static <U> ExplicitThreadLocal<U> withInitial(Supplier<U> initialValue) {
      return new ExplicitThreadLocal<U>() {
        @Override
        protected U initialValue() {
          return initialValue.get();
        }
      };
    }
  }

  /**
   * Bring-your-own ExecutorService graph builder.
   *
   * <p>Reads all the vectors from two copies of a {@link RandomAccessVectorValues}. Providing two
   * copies enables efficient retrieval without extra data copying, while avoiding collision of the
   * returned values.
   *
   * @param vectorsToAdd the vectors for which to build a nearest neighbors graph. Must be an
   *     independent accessor for the vectors
   * @param pool The ExecutorService to use. Must be an instance of ThreadPoolExecutor.
   * @param concurrentTasks the number of tasks to submit in parallel.
   */
  public Future<ConcurrentOnHeapHnswGraph> buildAsync(
      RandomAccessVectorValues<T> vectorsToAdd, ExecutorService pool, int concurrentTasks) {
    if (vectorsToAdd == this.vectors) {
      throw new IllegalArgumentException(
          "Vectors to build must be independent of the source of vectors provided to HnswGraphBuilder()");
    }
    if (infoStream.isEnabled(HNSW_COMPONENT)) {
      infoStream.message(HNSW_COMPONENT, "build graph from " + vectorsToAdd.size() + " vectors");
    }
    return addVectors(vectorsToAdd, pool, concurrentTasks);
  }

  // the goal here is to keep all the ExecutorService threads busy, but not to create potentially
  // millions of futures by naively throwing everything at submit at once.  So, we use
  // a semaphore to wait until a thread is free before adding a new task.
  private Future<ConcurrentOnHeapHnswGraph> addVectors(
      RandomAccessVectorValues<T> vectorsToAdd, ExecutorService pool, int concurrentTasks) {
    Semaphore semaphore = new Semaphore(concurrentTasks);
    Set<Integer> inFlight = ConcurrentHashMap.newKeySet();
    AtomicReference<Throwable> asyncException = new AtomicReference<>(null);

    ExplicitThreadLocal<RandomAccessVectorValues<T>> threadSafeVectors =
        createThreadSafeVectors(vectorsToAdd);

    for (int i = 0; i < vectorsToAdd.size(); i++) {
      final int node = i; // copy for closure
      try {
        semaphore.acquire();
        inFlight.add(node);
        pool.submit(
            () -> {
              try {
                addGraphNode(node, threadSafeVectors.get());
              } catch (Throwable e) {
                asyncException.set(e);
              } finally {
                semaphore.release();
                inFlight.remove(node);
              }
            });
      } catch (InterruptedException e) {
        throw new ThreadInterruptedException(e);
      }
    }

    // return a future that will complete when the inflight set is empty
    return CompletableFuture.supplyAsync(
        () -> {
          while (!inFlight.isEmpty()) {
            try {
              TimeUnit.MILLISECONDS.sleep(10);
            } catch (InterruptedException e) {
              throw new ThreadInterruptedException(e);
            }
          }
          if (asyncException.get() != null) {
            throw new CompletionException(asyncException.get());
          }
          hnsw.validateEntryNode();
          return hnsw;
        });
  }

  private static <T> ExplicitThreadLocal<RandomAccessVectorValues<T>> createThreadSafeVectors(
      RandomAccessVectorValues<T> vectorValues) {
    return ExplicitThreadLocal.withInitial(
        () -> {
          try {
            return vectorValues.copy();
          } catch (IOException e) {
            throw new UncheckedIOException(e);
          }
        });
  }

  /**
   * Adds a node to the graph, with the vector at the same ordinal in the given provider.
   *
   * <p>See {@link #addGraphNode(int, Object)} for more details.
   */
  public long addGraphNode(int node, RandomAccessVectorValues<T> values) throws IOException {
    return addGraphNode(node, values.vectorValue(node));
  }

  /** Set info-stream to output debugging information * */
  public void setInfoStream(InfoStream infoStream) {
    this.infoStream = infoStream;
  }

  public ConcurrentOnHeapHnswGraph getGraph() {
    return hnsw;
  }

  /** Number of inserts in progress, across all threads. */
  public int insertsInProgress() {
    return insertionsInProgress.size();
  }

  /**
   * Inserts a doc with vector value to the graph.
   *
   * <p>To allow correctness under concurrency, we track in-progress updates in a
   * ConcurrentSkipListSet. After adding ourselves, we take a snapshot of this set, and consider all
   * other in-progress updates as neighbor candidates (subject to normal level constraints).
   *
   * @param node the node ID to add
   * @param value the vector value to add
   * @return an estimate of the number of extra bytes used by the graph after adding the given node
   */
  public long addGraphNode(int node, T value) throws IOException {
    // do this before adding to in-progress, so a concurrent writer checking
    // the in-progress set doesn't have to worry about uninitialized neighbor sets
    final int nodeLevel = getRandomGraphLevel(ml);
    for (int level = nodeLevel; level >= 0; level--) {
      hnsw.addNode(level, node);
    }

    HnswGraph consistentView = hnsw.getView();
    NodeAtLevel progressMarker = new NodeAtLevel(nodeLevel, node);
    insertionsInProgress.add(progressMarker);
    ConcurrentSkipListSet<NodeAtLevel> inProgressBefore = insertionsInProgress.clone();
    try {
      // find ANN of the new node by searching the graph
      NodeAtLevel entry = hnsw.entry();
      int ep = entry.node;
      int[] eps = ep >= 0 ? new int[] {ep} : new int[0];
      var gs = graphSearcher.get();

      // for levels > nodeLevel search with topk = 1
      NeighborQueue candidates = new NeighborQueue(1, false);
      for (int level = entry.level; level > nodeLevel; level--) {
        candidates.clear();
        gs.searchLevel(
            candidates,
            value,
            1,
            level,
            eps,
            vectors.get(),
            consistentView,
            null,
            Integer.MAX_VALUE);
        eps = new int[] {candidates.pop()};
      }

      // for levels <= nodeLevel search with topk = beamWidth, and add connections
      candidates = beamCandidates.get();
      for (int level = Math.min(nodeLevel, entry.level); level >= 0; level--) {
        candidates.clear();
        // find best "natural" candidates at this level with a beam search
        gs.searchLevel(
            candidates,
            value,
            beamWidth,
            level,
            eps,
            vectors.get(),
            consistentView,
            null,
            Integer.MAX_VALUE);
        eps = candidates.nodes();

        // Update entry points and neighbors with these candidates.
        //
        // Note: We don't want to over-prune the neighbors, which can
        // happen if we group the concurrent candidates and the natural candidates together.
        //
        // Consider the following graph with "circular" test vectors:
        //
        // 0 -> 1
        // 1 <- 0
        // At this point we insert nodes 2 and 3 concurrently, denoted T1 and T2 for threads 1 and 2
        //   T1  T2
        //       insert 2 to L1 [2 is marked "in progress"]
        //   insert 3 to L1
        //   3 considers as neighbors 0, 1, 2; 0 and 1 are not diverse wrt 2
        // 3 -> 2 is added to graph
        //   3 is marked entry node
        //        2 follows 3 to L0, where 3 only has 2 as a neighbor
        // 2 -> 3 is added to graph
        // all further nodes will only be added to the 2/3 subgraph; 0/1 are partitioned forever
        //
        // Considering concurrent inserts separately from natural candidates solves this problem;
        // both 1 and 2 will be added as neighbors to 3, avoiding the partition, and 2 will then
        // pick up the connection to 1 that it's supposed to have as well.
        addForwardLinks(level, node, candidates); // natural candidates
        addForwardLinks(level, node, inProgressBefore, progressMarker); // concurrent candidates
        // Backlinking is the same for both natural and concurrent candidates.
        addBackLinks(level, node);
      }

      // If we're being added in a new level above the entry point, consider concurrent insertions
      // for inclusion as neighbors at that level. There are no natural neighbors yet.
      for (int level = entry.level + 1; level <= nodeLevel; level++) {
        addForwardLinks(level, node, inProgressBefore, progressMarker);
        addBackLinks(level, node);
      }

      hnsw.markComplete(nodeLevel, node);
    } finally {
      insertionsInProgress.remove(progressMarker);
    }

    return hnsw.ramBytesUsedOneNode(nodeLevel);
  }

  private void addForwardLinks(int level, int newNode, NeighborQueue candidates) {
    NeighborArray scratch = popToScratch(candidates); // worst are first
    ConcurrentNeighborSet neighbors = hnsw.getNeighbors(level, newNode);
    neighbors.insertDiverse(scratch);
  }

  private void addForwardLinks(
      int level, int newNode, Set<NodeAtLevel> inProgress, NodeAtLevel progressMarker)
      throws IOException {
    if (inProgress.isEmpty()) {
      return;
    }

    T v = vectors.get().vectorValue(newNode);
    NeighborQueue candidates = new NeighborQueue(inProgress.size(), false);
    for (NodeAtLevel n : inProgress) {
      if (n.level >= level && n != progressMarker) {
        candidates.add(n.node, scoreBetween(v, vectorsCopy.get().vectorValue(n.node)));
      }
    }
    ConcurrentNeighborSet neighbors = hnsw.getNeighbors(level, newNode);
    NeighborArray scratch = popToScratch(candidates); // worst are first
    neighbors.insertDiverse(scratch);
  }

  private void addBackLinks(int level, int newNode) throws IOException {
    ConcurrentNeighborSet neighbors = hnsw.getNeighbors(level, newNode);
    neighbors.backlink(i -> hnsw.getNeighbors(level, i));
  }

  protected float scoreBetween(T v1, T v2) {
    return switch (vectorEncoding) {
      case BYTE -> similarityFunction.compare((byte[]) v1, (byte[]) v2);
      case FLOAT32 -> similarityFunction.compare((float[]) v1, (float[]) v2);
    };
  }

  private NeighborArray popToScratch(NeighborQueue candidates) {
    NeighborArray scratch = this.scratchNeighbors.get();
    scratch.clear();
    int candidateCount = candidates.size();
    // extract all the Neighbors from the queue into an array; these will now be
    // sorted from worst to best
    for (int i = 0; i < candidateCount; i++) {
      float maxSimilarity = candidates.topScore();
      scratch.addInOrder(candidates.pop(), maxSimilarity);
    }
    return scratch;
  }

  int getRandomGraphLevel(double ml) {
    double randDouble;
    do {
      randDouble =
          ThreadLocalRandom.current().nextDouble(); // avoid 0 value, as log(0) is undefined
    } while (randDouble == 0.0);
    return ((int) (-log(randDouble) * ml));
  }
}
