/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.lucene.util.hnsw.math.ml.neuralnet.twod;

import java.util.List;
import java.util.ArrayList;
import java.util.Iterator;
import java.io.Serializable;
import java.io.ObjectInputStream;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.Neuron;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.Network;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.FeatureInitializer;
import org.apache.lucene.util.hnsw.math.ml.neuralnet.SquareNeighbourhood;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooSmallException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.MathInternalError;


public class NeuronSquareMesh2D
    implements Iterable<Neuron>,
               Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final Network network;
    
    private final int numberOfRows;
    
    private final int numberOfColumns;
    
    private final boolean wrapRows;
    
    private final boolean wrapColumns;
    
    private final SquareNeighbourhood neighbourhood;
    
    private final long[][] identifiers;

    
    public enum HorizontalDirection {
        
       RIGHT,
       
       CENTER,
       
       LEFT,
    }
    
    public enum VerticalDirection {
        
        UP,
        
        CENTER,
        
        DOWN,
    }

    
    NeuronSquareMesh2D(boolean wrapRowDim,
                       boolean wrapColDim,
                       SquareNeighbourhood neighbourhoodType,
                       double[][][] featuresList) {
        numberOfRows = featuresList.length;
        numberOfColumns = featuresList[0].length;

        if (numberOfRows < 2) {
            throw new NumberIsTooSmallException(numberOfRows, 2, true);
        }
        if (numberOfColumns < 2) {
            throw new NumberIsTooSmallException(numberOfColumns, 2, true);
        }

        wrapRows = wrapRowDim;
        wrapColumns = wrapColDim;
        neighbourhood = neighbourhoodType;

        final int fLen = featuresList[0][0].length;
        network = new Network(0, fLen);
        identifiers = new long[numberOfRows][numberOfColumns];

        // Add neurons.
        for (int i = 0; i < numberOfRows; i++) {
            for (int j = 0; j < numberOfColumns; j++) {
                identifiers[i][j] = network.createNeuron(featuresList[i][j]);
            }
        }

        // Add links.
        createLinks();
    }

    
    public NeuronSquareMesh2D(int numRows,
                              boolean wrapRowDim,
                              int numCols,
                              boolean wrapColDim,
                              SquareNeighbourhood neighbourhoodType,
                              FeatureInitializer[] featureInit) {
        if (numRows < 2) {
            throw new NumberIsTooSmallException(numRows, 2, true);
        }
        if (numCols < 2) {
            throw new NumberIsTooSmallException(numCols, 2, true);
        }

        numberOfRows = numRows;
        wrapRows = wrapRowDim;
        numberOfColumns = numCols;
        wrapColumns = wrapColDim;
        neighbourhood = neighbourhoodType;
        identifiers = new long[numberOfRows][numberOfColumns];

        final int fLen = featureInit.length;
        network = new Network(0, fLen);

        // Add neurons.
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                final double[] features = new double[fLen];
                for (int fIndex = 0; fIndex < fLen; fIndex++) {
                    features[fIndex] = featureInit[fIndex].value();
                }
                identifiers[i][j] = network.createNeuron(features);
            }
        }

        // Add links.
        createLinks();
    }

    
    private NeuronSquareMesh2D(boolean wrapRowDim,
                               boolean wrapColDim,
                               SquareNeighbourhood neighbourhoodType,
                               Network net,
                               long[][] idGrid) {
        numberOfRows = idGrid.length;
        numberOfColumns = idGrid[0].length;
        wrapRows = wrapRowDim;
        wrapColumns = wrapColDim;
        neighbourhood = neighbourhoodType;
        network = net;
        identifiers = idGrid;
    }

    
    public synchronized NeuronSquareMesh2D copy() {
        final long[][] idGrid = new long[numberOfRows][numberOfColumns];
        for (int r = 0; r < numberOfRows; r++) {
            for (int c = 0; c < numberOfColumns; c++) {
                idGrid[r][c] = identifiers[r][c];
            }
        }

        return new NeuronSquareMesh2D(wrapRows,
                                      wrapColumns,
                                      neighbourhood,
                                      network.copy(),
                                      idGrid);
    }

    
    public Iterator<Neuron> iterator() {
        return network.iterator();
    }

    
    public Network getNetwork() {
        return network;
    }

    
    public int getNumberOfRows() {
        return numberOfRows;
    }

    
    public int getNumberOfColumns() {
        return numberOfColumns;
    }

    
    public Neuron getNeuron(int i,
                            int j) {
        if (i < 0 ||
            i >= numberOfRows) {
            throw new OutOfRangeException(i, 0, numberOfRows - 1);
        }
        if (j < 0 ||
            j >= numberOfColumns) {
            throw new OutOfRangeException(j, 0, numberOfColumns - 1);
        }

        return network.getNeuron(identifiers[i][j]);
    }

    
    public Neuron getNeuron(int row,
                            int col,
                            HorizontalDirection alongRowDir,
                            VerticalDirection alongColDir) {
        final int[] location = getLocation(row, col, alongRowDir, alongColDir);

        return location == null ? null : getNeuron(location[0], location[1]);
    }

    
    private int[] getLocation(int row,
                              int col,
                              HorizontalDirection alongRowDir,
                              VerticalDirection alongColDir) {
        final int colOffset;
        switch (alongRowDir) {
        case LEFT:
            colOffset = -1;
            break;
        case RIGHT:
            colOffset = 1;
            break;
        case CENTER:
            colOffset = 0;
            break;
        default:
            // Should never happen.
            throw new MathInternalError();
        }
        int colIndex = col + colOffset;
        if (wrapColumns) {
            if (colIndex < 0) {
                colIndex += numberOfColumns;
            } else {
                colIndex %= numberOfColumns;
            }
        }

        final int rowOffset;
        switch (alongColDir) {
        case UP:
            rowOffset = -1;
            break;
        case DOWN:
            rowOffset = 1;
            break;
        case CENTER:
            rowOffset = 0;
            break;
        default:
            // Should never happen.
            throw new MathInternalError();
        }
        int rowIndex = row + rowOffset;
        if (wrapRows) {
            if (rowIndex < 0) {
                rowIndex += numberOfRows;
            } else {
                rowIndex %= numberOfRows;
            }
        }

        if (rowIndex < 0 ||
            rowIndex >= numberOfRows ||
            colIndex < 0 ||
            colIndex >= numberOfColumns) {
            return null;
        } else {
            return new int[] { rowIndex, colIndex };
        }
    }

    
    private void createLinks() {
        // "linkEnd" will store the identifiers of the "neighbours".
        final List<Long> linkEnd = new ArrayList<Long>();
        final int iLast = numberOfRows - 1;
        final int jLast = numberOfColumns - 1;
        for (int i = 0; i < numberOfRows; i++) {
            for (int j = 0; j < numberOfColumns; j++) {
                linkEnd.clear();

                switch (neighbourhood) {

                case MOORE:
                    // Add links to "diagonal" neighbours.
                    if (i > 0) {
                        if (j > 0) {
                            linkEnd.add(identifiers[i - 1][j - 1]);
                        }
                        if (j < jLast) {
                            linkEnd.add(identifiers[i - 1][j + 1]);
                        }
                    }
                    if (i < iLast) {
                        if (j > 0) {
                            linkEnd.add(identifiers[i + 1][j - 1]);
                        }
                        if (j < jLast) {
                            linkEnd.add(identifiers[i + 1][j + 1]);
                        }
                    }
                    if (wrapRows) {
                        if (i == 0) {
                            if (j > 0) {
                                linkEnd.add(identifiers[iLast][j - 1]);
                            }
                            if (j < jLast) {
                                linkEnd.add(identifiers[iLast][j + 1]);
                            }
                        } else if (i == iLast) {
                            if (j > 0) {
                                linkEnd.add(identifiers[0][j - 1]);
                            }
                            if (j < jLast) {
                                linkEnd.add(identifiers[0][j + 1]);
                            }
                        }
                    }
                    if (wrapColumns) {
                        if (j == 0) {
                            if (i > 0) {
                                linkEnd.add(identifiers[i - 1][jLast]);
                            }
                            if (i < iLast) {
                                linkEnd.add(identifiers[i + 1][jLast]);
                            }
                        } else if (j == jLast) {
                             if (i > 0) {
                                 linkEnd.add(identifiers[i - 1][0]);
                             }
                             if (i < iLast) {
                                 linkEnd.add(identifiers[i + 1][0]);
                             }
                        }
                    }
                    if (wrapRows &&
                        wrapColumns) {
                        if (i == 0 &&
                            j == 0) {
                            linkEnd.add(identifiers[iLast][jLast]);
                        } else if (i == 0 &&
                                   j == jLast) {
                            linkEnd.add(identifiers[iLast][0]);
                        } else if (i == iLast &&
                                   j == 0) {
                            linkEnd.add(identifiers[0][jLast]);
                        } else if (i == iLast &&
                                   j == jLast) {
                            linkEnd.add(identifiers[0][0]);
                        }
                    }

                    // Case falls through since the "Moore" neighbourhood
                    // also contains the neurons that belong to the "Von
                    // Neumann" neighbourhood.

                    // fallthru (CheckStyle)
                case VON_NEUMANN:
                    // Links to preceding and following "row".
                    if (i > 0) {
                        linkEnd.add(identifiers[i - 1][j]);
                    }
                    if (i < iLast) {
                        linkEnd.add(identifiers[i + 1][j]);
                    }
                    if (wrapRows) {
                        if (i == 0) {
                            linkEnd.add(identifiers[iLast][j]);
                        } else if (i == iLast) {
                            linkEnd.add(identifiers[0][j]);
                        }
                    }

                    // Links to preceding and following "column".
                    if (j > 0) {
                        linkEnd.add(identifiers[i][j - 1]);
                    }
                    if (j < jLast) {
                        linkEnd.add(identifiers[i][j + 1]);
                    }
                    if (wrapColumns) {
                        if (j == 0) {
                            linkEnd.add(identifiers[i][jLast]);
                        } else if (j == jLast) {
                            linkEnd.add(identifiers[i][0]);
                        }
                    }
                    break;

                default:
                    throw new MathInternalError(); // Cannot happen.
                }

                final Neuron aNeuron = network.getNeuron(identifiers[i][j]);
                for (long b : linkEnd) {
                    final Neuron bNeuron = network.getNeuron(b);
                    // Link to all neighbours.
                    // The reverse links will be added as the loop proceeds.
                    network.addLink(aNeuron, bNeuron);
                }
            }
        }
    }

    
    private void readObject(ObjectInputStream in) {
        throw new IllegalStateException();
    }

    
    private Object writeReplace() {
        final double[][][] featuresList = new double[numberOfRows][numberOfColumns][];
        for (int i = 0; i < numberOfRows; i++) {
            for (int j = 0; j < numberOfColumns; j++) {
                featuresList[i][j] = getNeuron(i, j).getFeatures();
            }
        }

        return new SerializationProxy(wrapRows,
                                      wrapColumns,
                                      neighbourhood,
                                      featuresList);
    }

    
    private static class SerializationProxy implements Serializable {
        
        private static final long serialVersionUID = 20130226L;
        
        private final boolean wrapRows;
        
        private final boolean wrapColumns;
        
        private final SquareNeighbourhood neighbourhood;
        
        private final double[][][] featuresList;

        
        SerializationProxy(boolean wrapRows,
                           boolean wrapColumns,
                           SquareNeighbourhood neighbourhood,
                           double[][][] featuresList) {
            this.wrapRows = wrapRows;
            this.wrapColumns = wrapColumns;
            this.neighbourhood = neighbourhood;
            this.featuresList = featuresList;
        }

        
        private Object readResolve() {
            return new NeuronSquareMesh2D(wrapRows,
                                          wrapColumns,
                                          neighbourhood,
                                          featuresList);
        }
    }
}
