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
import java.util.Arrays;
import java.util.Collections;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListSet;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.AtomicBitSet;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.NamedThreadFactory;
import org.apache.lucene.util.ThreadInterruptedException;
import org.apache.lucene.util.hnsw.ConcurrentOnHeapHnswGraph.NodeAtLevel;

/**
 * Builder for Concurrent HNSW graph. See {@link HnswGraph} for a high level overview, and the
 * comments to `addGraphNode` for details on the concurrent building approach.
 *
 * @param <T> the type of vector
 */
public class ConcurrentHnswGraphBuilder<T> {
  protected static final Log LOG = new Log();

  /** Default number of maximum connections per node */
  public static final int DEFAULT_MAX_CONN = 16;

  /**
   * Default number of the size of the queue maintained while searching during a graph construction.
   */
  public static final int DEFAULT_BEAM_WIDTH = 100;

  /** A name for the HNSW component for the info-stream * */
  public static final String HNSW_COMPONENT = "HNSW";

  private final int beamWidth;
  private final double ml;
  private final ExplicitThreadLocal<NeighborArray> scratchNeighbors;

  private final VectorSimilarityFunction similarityFunction;
  private final VectorEncoding vectorEncoding;
  private final RandomAccessVectorValues<T> vectors;
  private final ExplicitThreadLocal<HnswGraphSearcher<T>> graphSearcher;

  final ConcurrentOnHeapHnswGraph hnsw;
  private final ConcurrentSkipListSet<NodeAtLevel> insertionsInProgress =
      new ConcurrentSkipListSet<>();

  private InfoStream infoStream = InfoStream.getDefault();

  // we need two sources of vectors in order to perform diversity check comparisons without
  // colliding
  private final RandomAccessVectorValues<T> vectorsCopy;

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
   * @param vectors the vectors whose relations are represented by the graph - must provide a
   *     different view over those vectors than the one used to add via addGraphNode.
   * @param M – graph fanout parameter used to calculate the maximum number of connections a node
   *     can have – M on upper layers, and M * 2 on the lowest level.
   * @param beamWidth the size of the beam search to use when finding nearest neighbors.
   */
  public ConcurrentHnswGraphBuilder(
      RandomAccessVectorValues<T> vectors,
      VectorEncoding vectorEncoding,
      VectorSimilarityFunction similarityFunction,
      int M,
      int beamWidth)
      throws IOException {
    this.vectors = vectors;
    this.vectorsCopy = vectors.copy();
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
    this.hnsw = new ConcurrentOnHeapHnswGraph(M);
    this.graphSearcher =
        ExplicitThreadLocal.withInitial(
            () -> {
              return new HnswGraphSearcher<>(
                  vectorEncoding,
                  similarityFunction,
                  new NeighborQueue(beamWidth, true),
                  new AtomicBitSet(this.vectors.size()));
            });
    // in scratch we store candidates in reverse order: worse candidates are first
    scratchNeighbors =
        ExplicitThreadLocal.withInitial(() -> new NeighborArray(Math.max(beamWidth, M + 1), false));
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
   * Reads all the vectors from two copies of a {@link RandomAccessVectorValues}. Providing two
   * copies enables efficient retrieval without extra data copying, while avoiding collision of the
   * returned values.
   *
   * @param vectorsToAdd the vectors for which to build a nearest neighbors graph. Must be an
   *     independent accessor for the vectors
   * @param autoParallel if true, the builder will allocate one thread per core to building the
   *     graph; if false, it will use a single thread. For more fine-grained control, use the
   *     ExecutorService (ThreadPoolExecutor) overload.
   */
  public ConcurrentOnHeapHnswGraph build(
      RandomAccessVectorValues<T> vectorsToAdd, boolean autoParallel) throws IOException {
    ExecutorService es;
    if (autoParallel) {
      es =
          Executors.newFixedThreadPool(
              Runtime.getRuntime().availableProcessors(),
              new NamedThreadFactory("Concurrent HNSW builder"));
    } else {
      es = Executors.newSingleThreadExecutor(new NamedThreadFactory("Concurrent HNSW builder"));
    }

    Future<ConcurrentOnHeapHnswGraph> f = buildAsync(vectorsToAdd, es);
    try {
      return f.get();
    } catch (InterruptedException e) {
      throw new ThreadInterruptedException(e);
    } catch (ExecutionException e) {
      throw new IOException(e);
    } finally {
      es.shutdown();
    }
  }

  public ConcurrentOnHeapHnswGraph build(RandomAccessVectorValues<T> vectorsToAdd)
      throws IOException {
    return build(vectorsToAdd, true);
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
   */
  public Future<ConcurrentOnHeapHnswGraph> buildAsync(
      RandomAccessVectorValues<T> vectorsToAdd, ExecutorService pool) {
    if (vectorsToAdd == this.vectors) {
      throw new IllegalArgumentException(
          "Vectors to build must be independent of the source of vectors provided to HnswGraphBuilder()");
    }
    if (infoStream.isEnabled(HNSW_COMPONENT)) {
      infoStream.message(HNSW_COMPONENT, "build graph from " + vectorsToAdd.size() + " vectors");
    }
    if (!(pool instanceof ThreadPoolExecutor)) {
      throw new IllegalArgumentException("ExecutorService must be a ThreadPoolExecutor");
    }
    return addVectors(vectorsToAdd, (ThreadPoolExecutor) pool);
  }

  // the goal here is to keep all the ExecutorService threads busy, but not to create potentially
  // millions of futures by naively throwing everything at submit at once.  So, we use
  // a semaphore to wait until a thread is free before adding a new task.
  private Future<ConcurrentOnHeapHnswGraph> addVectors(
      RandomAccessVectorValues<T> vectorsToAdd, ThreadPoolExecutor pool) {
    Semaphore semaphore = new Semaphore(pool.getMaximumPoolSize());
    Set<Integer> inFlight = ConcurrentHashMap.newKeySet();
    AtomicReference<Throwable> asyncException = new AtomicReference<>(null);

    for (int i = 0; i < vectorsToAdd.size(); i++) {
      final int node = i; // copy for closure
      try {
        semaphore.acquire();
        inFlight.add(node);
        pool.submit(
            () -> {
              try {
                addGraphNode(node, vectorsToAdd);
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

  public void addGraphNode(int node, RandomAccessVectorValues<T> values) throws IOException {
    addGraphNode(node, values.vectorValue(node));
  }

  /** Set info-stream to output debugging information * */
  public void setInfoStream(InfoStream infoStream) {
    this.infoStream = infoStream;
  }

  public ConcurrentOnHeapHnswGraph getGraph() {
    return hnsw;
  }

  /**
   * Inserts a doc with vector value to the graph.
   *
   * <p>To allow correctness under concurrency, we track in-progress updates in a ConcurrentSkipListSet. After adding ourselves, we take a snapshot of this set, and consider all
   * other in-progress updates as neighbor candidates (subject to normal level constraints).
   */
  public void addGraphNode(int node, T value) throws IOException {
    // do this before adding to in-progress, so a concurrent writer checking
    // the in-progress set doesn't have to worry about uninitialized neighbor sets
    final int nodeLevel = getRandomGraphLevel(ml);
    LOG.info(String.format("%s adding node %s at level %s", Thread.currentThread().getId(), node, nodeLevel));
    for (int level = nodeLevel; level >= 0; level--) {
      hnsw.addNode(level, node);
    }

    NodeAtLevel progressMarker = new NodeAtLevel(nodeLevel, node);
    insertionsInProgress.add(progressMarker);
    ConcurrentSkipListSet<NodeAtLevel> inProgressBefore = insertionsInProgress.clone();
    try {
      // find ANN of the new node by searching the graph
      NodeAtLevel entry = hnsw.entry();
      int ep = entry.node;
      int[] eps = ep >= 0 ? new int[] {ep} : new int[0];
      NeighborQueue candidates;

      // if we're being added in a new level above the entry point, add any concurrent insertions
      // as neighbors at that level
      for (int level = nodeLevel; level > entry.level; level--) {
        candidates = new NeighborQueue(inProgressBefore.size(), false);
        for (NodeAtLevel n : inProgressBefore) {
          if (n.level >= level && n != progressMarker) {
            candidates.add(n.node, scoreBetween(n.node, node));
          }
        }

        LOG.info(String.format("%s L%s (above entry) adding in-progress candidates %s", Thread.currentThread().getId(), level, candidates));
        addDiverseNeighbors(level, node, candidates, Stream.empty());
      }

      // TODO another problem:
      // Level 1 consists of node 9.
      // T1: Add 24 at L1.
      // T1: wires 24 to 9 at L1 and vice versa.  24 now exists in L1 but not L0.
      // T2: Add 23 at L0
      // T2: follows index from 9 -> 24
      // T2: 24 @L0 has no neighbors
      //     -> 23 gets wired to 24 alone instead of its correct neighbors

      // check in-progress candidates for currMaxLevel and below
      // for levels > nodeLevel search with topk = 1
      String entryString = Arrays.toString(eps);
      for (int level = entry.level; level > nodeLevel; level--) {
        candidates = graphSearcher.get().searchLevel(value, 1, level, eps, vectors, hnsw.getView());
        eps = new int[] {candidates.pop()};
        entryString += " -> " + Arrays.toString(eps);
      }
      if (nodeLevel != entry.level) {
        LOG.info(String.format("%s entry path is %s", Thread.currentThread().getId(), entryString));
      }
      // for levels <= nodeLevel search with topk = beamWidth, and add connections
      for (int level = Math.min(nodeLevel, entry.level); level >= 0; level--) {
        // find best candidates at this level with a beam search
        candidates =
            graphSearcher.get().searchLevel(value, beamWidth, level, eps, vectors, hnsw.getView());
        LOG.info(String.format("%s L%s initial candidates from %s are %s", Thread.currentThread().getId(), level, Arrays.toString(eps), Arrays.toString(candidates.nodes())));
        // any nodes that are being added concurrently at this level are also candidates
        int thisLevel = level;
        Stream<Integer> concurrentCandidates = inProgressBefore
            .stream()
            .filter(n -> n.level >= thisLevel && n != progressMarker)
            .map(n -> n.node);
        LOG.info(String.format("%s    concurrent additions are %s", Thread.currentThread().getId(), inProgressBefore));
        // update entry points and neighbors with these candidates
        eps = candidates.nodes();
        addDiverseNeighbors(level, node, candidates, concurrentCandidates);
      }

      // update entry node last, once everything is wired together
      hnsw.maybeUpdateEntryNode(nodeLevel, node);
    } finally {
      LOG.info(String.format("%s finished adding node %s", Thread.currentThread().getId(), node));
      insertionsInProgress.remove(progressMarker);
    }
  }

  /**
   * Add links from new node -> candidates.
   * See ConcurrentNeighborSet for an explanation of "diverse."
  */
  private void addDiverseNeighbors(int level, int newNode, NeighborQueue candidates, Stream<Integer> concurrentCandidates)
      throws IOException {
    // the concurrency here is a bit tricky.  The initial implementation merged concurrent
    // inserts into the candidate set, but this is incorrect.  Consider the following graph
    // with the "circular" test vectors:
    //
    // 0 -> 1
    // 1 <- 0
    // at this point we insert nodes 2 and 3 concurrently:
    //     insert 2 to L0 [2 is marked "in progress"]
    //   insert 3 to L1
    //   insert 3 to L0
    //   3 considers as neighbors 0, 1, 2; 0 and 1 are not diverse wrt 2
    // 3 -> 2 is added to graph
    // [3 is now entry point]
    //     2 follows 3 to L0, where 3 only has 2 as a neighbor
    // 2 -> 3 is added to graph
    // all further nodes will only be added to the 2/3 subgraph; 0/1 are partitioned forever
    //
    // I can't think of a reasonable way to make the entry point "transactional" wrt the
    // neighbor graph, so we have to be a bit more conservative.
    //
    // Instead, we first consider candidates that are fully inserted in the graph for the
    // diversity heuristic, and then we add links to all concurrent insertions to ensure
    // connectivity.  (CNS.insert will take care of pruning back if we hit our max neighbor
    // count.)
    ConcurrentNeighborSet neighbors = hnsw.getNeighbors(level, newNode);
    NeighborArray scratch = popToScratch(candidates); // worst are first
    LOG.info(String.format("%s    pre-diverse neighbors are %s",
        Thread.currentThread().getId(), Arrays.toString(Arrays.copyOf(scratch.node(), scratch.size()))));
    // TODO we still have a problem here, because insertDiverse compares the nodes being inserted to all
    // existing neighbors, so if other nodes insert backlings to us before we call this, we could end up
    // having potentially all of them pruned.
    // Solution is to only consider diversity among the set being inserted.
    // Example at right -- 24 gets inserted, [correctly] considered 0-22 as neighbors, but 23, 25, 26 are
    // inserted concurrently and none of 0-22 are added.
    neighbors.insertDiverse(scratch, this::scoreBetween);
    LOG.info(String.format("%s    post-diverse neighbors are %s",
        Thread.currentThread().getId(), neighbors.neighborsAsList()));

    concurrentCandidates.forEach(nbr -> {
      try {
        neighbors.insert(nbr, scoreBetween(nbr, newNode), this::scoreBetween);
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    });
    LOG.info(String.format("%s    neighbors after adding concurrent inserts are %s",
        Thread.currentThread().getId(), neighbors.neighborsAsList()));

    // Add backlinks from neighbors -> new node (again applying diversity heuristic)
    neighbors.forEach(
        (nbr, nbrScore) -> {
          ConcurrentNeighborSet nbrNbr = hnsw.getNeighbors(level, nbr);
          nbrNbr.insert(newNode, nbrScore, this::scoreBetween);
        });
  }

  private float scoreBetween(int i, int j) throws IOException {
    final T v1 = vectorsCopy.vectorValue(i);
    final T v2 = vectorsCopy.vectorValue(j);
    return scoreBetween(v1, v2);
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
      scratch.add(candidates.pop(), maxSimilarity);
    }
    return scratch;
  }

  private static int getRandomGraphLevel(double ml) {
    double randDouble;
    do {
      randDouble =
          ThreadLocalRandom.current().nextDouble(); // avoid 0 value, as log(0) is undefined
    } while (randDouble == 0.0);
    return ((int) (-log(randDouble) * ml));
  }
}
