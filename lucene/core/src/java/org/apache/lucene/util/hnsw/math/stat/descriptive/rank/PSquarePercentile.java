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
package org.apache.lucene.util.hnsw.math.stat.descriptive.rank;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.Serializable;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.analysis.interpolation.LinearInterpolator;
import org.apache.lucene.util.hnsw.math.analysis.interpolation.NevilleInterpolator;
import org.apache.lucene.util.hnsw.math.analysis.interpolation.UnivariateInterpolator;
import org.apache.lucene.util.hnsw.math.exception.InsufficientDataException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.stat.descriptive.AbstractStorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.stat.descriptive.StorelessUnivariateStatistic;
import org.apache.lucene.util.hnsw.math.util.MathArrays;
import org.apache.lucene.util.hnsw.math.util.MathUtils;
import org.apache.lucene.util.hnsw.math.util.Precision;


public class PSquarePercentile extends AbstractStorelessUnivariateStatistic
        implements StorelessUnivariateStatistic, Serializable {

    
    private static final int PSQUARE_CONSTANT = 5;

    
    private static final double DEFAULT_QUANTILE_DESIRED = 50d;

    
    private static final long serialVersionUID = 2283912083175715479L;

    
    private static final DecimalFormat DECIMAL_FORMAT = new DecimalFormat(
            "00.00");

    
    private final List<Double> initialFive = new FixedCapacityList<Double>(
            PSQUARE_CONSTANT);

    
    private final double quantile;

    
    private transient double lastObservation;

    
    private PSquareMarkers markers = null;

    
    private double pValue = Double.NaN;

    
    private long countOfObservations;

    
    public PSquarePercentile(final double p) {
        if (p > 100 || p < 0) {
            throw new OutOfRangeException(LocalizedFormats.OUT_OF_RANGE,
                    p, 0, 100);
        }
        this.quantile = p / 100d;// always set it within (0,1]
    }

    
    PSquarePercentile() {
        this(DEFAULT_QUANTILE_DESIRED);
    }

    
    @Override
    public int hashCode() {
        double result = getResult();
        result = Double.isNaN(result) ? 37 : result;
        final double markersHash = markers == null ? 0 : markers.hashCode();
        final double[] toHash = {result, quantile, markersHash, countOfObservations};
        return Arrays.hashCode(toHash);
    }

    
    @Override
    public boolean equals(Object o) {
        boolean result = false;
        if (this == o) {
            result = true;
        } else if (o != null && o instanceof PSquarePercentile) {
            PSquarePercentile that = (PSquarePercentile) o;
            boolean isNotNull = markers != null && that.markers != null;
            boolean isNull = markers == null && that.markers == null;
            result = isNotNull ? markers.equals(that.markers) : isNull;
            // markers as in the case of first
            // five observations
            result = result && getN() == that.getN();
        }
        return result;
    }

    
    @Override
    public void increment(final double observation) {
        // Increment counter
        countOfObservations++;

        // Store last observation
        this.lastObservation = observation;

        // 0. Use Brute force for <5
        if (markers == null) {
            if (initialFive.add(observation)) {
                Collections.sort(initialFive);
                pValue =
                        initialFive
                                .get((int) (quantile * (initialFive.size() - 1)));
                return;
            }
            // 1. Initialize once after 5th observation
            markers = newMarkers(initialFive, quantile);
        }
        // 2. process a Data Point and return pValue
        pValue = markers.processDataPoint(observation);
    }

    
    @Override
    public String toString() {

        if (markers == null) {
            return String.format("obs=%s pValue=%s",
                    DECIMAL_FORMAT.format(lastObservation),
                    DECIMAL_FORMAT.format(pValue));
        } else {
            return String.format("obs=%s markers=%s",
                    DECIMAL_FORMAT.format(lastObservation), markers.toString());
        }
    }

    
    public long getN() {
        return countOfObservations;
    }

    
    @Override
    public StorelessUnivariateStatistic copy() {
        // multiply quantile by 100 now as anyway constructor divides it by 100
        PSquarePercentile copy = new PSquarePercentile(100d * quantile);

        if (markers != null) {
            copy.markers = (PSquareMarkers) markers.clone();
        }
        copy.countOfObservations = countOfObservations;
        copy.pValue = pValue;
        copy.initialFive.clear();
        copy.initialFive.addAll(initialFive);
        return copy;
    }

    
    public double quantile() {
        return quantile;
    }

    
    @Override
    public void clear() {
        markers = null;
        initialFive.clear();
        countOfObservations = 0L;
        pValue = Double.NaN;
    }

    
    @Override
    public double getResult() {
        if (Double.compare(quantile, 1d) == 0) {
            pValue = maximum();
        } else if (Double.compare(quantile, 0d) == 0) {
            pValue = minimum();
        }
        return pValue;
    }

    
    private double maximum() {
        double val = Double.NaN;
        if (markers != null) {
            val = markers.height(PSQUARE_CONSTANT);
        } else if (!initialFive.isEmpty()) {
            val = initialFive.get(initialFive.size() - 1);
        }
        return val;
    }

    
    private double minimum() {
        double val = Double.NaN;
        if (markers != null) {
            val = markers.height(1);
        } else if (!initialFive.isEmpty()) {
            val = initialFive.get(0);
        }
        return val;
    }

    
    private static class Markers implements PSquareMarkers, Serializable {
        
        private static final long serialVersionUID = 1L;

        
        private static final int LOW = 2;

        
        private static final int HIGH = 4;

        
        private final Marker[] markerArray;

        
        private transient int k = -1;

        
        private Markers(final Marker[] theMarkerArray) {
            MathUtils.checkNotNull(theMarkerArray);
            markerArray = theMarkerArray;
            for (int i = 1; i < PSQUARE_CONSTANT; i++) {
                markerArray[i].previous(markerArray[i - 1])
                        .next(markerArray[i + 1]).index(i);
            }
            markerArray[0].previous(markerArray[0]).next(markerArray[1])
                    .index(0);
            markerArray[5].previous(markerArray[4]).next(markerArray[5])
                    .index(5);
        }

        
        private Markers(final List<Double> initialFive, final double p) {
            this(createMarkerArray(initialFive, p));
        }

        
        private static Marker[] createMarkerArray(
                final List<Double> initialFive, final double p) {
            final int countObserved =
                    initialFive == null ? -1 : initialFive.size();
            if (countObserved < PSQUARE_CONSTANT) {
                throw new InsufficientDataException(
                        LocalizedFormats.INSUFFICIENT_OBSERVED_POINTS_IN_SAMPLE,
                        countObserved, PSQUARE_CONSTANT);
            }
            Collections.sort(initialFive);
            return new Marker[] {
                    new Marker(),// Null Marker
                    new Marker(initialFive.get(0), 1, 0, 1),
                    new Marker(initialFive.get(1), 1 + 2 * p, p / 2, 2),
                    new Marker(initialFive.get(2), 1 + 4 * p, p, 3),
                    new Marker(initialFive.get(3), 3 + 2 * p, (1 + p) / 2, 4),
                    new Marker(initialFive.get(4), 5, 1, 5) };
        }

        
        @Override
        public int hashCode() {
            return Arrays.deepHashCode(markerArray);
        }

        
        @Override
        public boolean equals(Object o) {
            boolean result = false;
            if (this == o) {
                result = true;
            } else if (o != null && o instanceof Markers) {
                Markers that = (Markers) o;
                result = Arrays.deepEquals(markerArray, that.markerArray);
            }
            return result;
        }

        
        public double processDataPoint(final double inputDataPoint) {

            // 1. Find cell and update minima and maxima
            final int kthCell = findCellAndUpdateMinMax(inputDataPoint);

            // 2. Increment positions
            incrementPositions(1, kthCell + 1, 5);

            // 2a. Update desired position with increments
            updateDesiredPositions();

            // 3. Adjust heights of m[2-4] if necessary
            adjustHeightsOfMarkers();

            // 4. Return percentile
            return getPercentileValue();
        }

        
        public double getPercentileValue() {
            return height(3);
        }

        
        private int findCellAndUpdateMinMax(final double observation) {
            k = -1;
            if (observation < height(1)) {
                markerArray[1].markerHeight = observation;
                k = 1;
            } else if (observation < height(2)) {
                k = 1;
            } else if (observation < height(3)) {
                k = 2;
            } else if (observation < height(4)) {
                k = 3;
            } else if (observation <= height(5)) {
                k = 4;
            } else {
                markerArray[5].markerHeight = observation;
                k = 4;
            }
            return k;
        }

        
        private void adjustHeightsOfMarkers() {
            for (int i = LOW; i <= HIGH; i++) {
                estimate(i);
            }
        }

        
        public double estimate(final int index) {
            if (index < LOW || index > HIGH) {
                throw new OutOfRangeException(index, LOW, HIGH);
            }
            return markerArray[index].estimate();
        }

        
        private void incrementPositions(final int d, final int startIndex,
                final int endIndex) {
            for (int i = startIndex; i <= endIndex; i++) {
                markerArray[i].incrementPosition(d);
            }
        }

        
        private void updateDesiredPositions() {
            for (int i = 1; i < markerArray.length; i++) {
                markerArray[i].updateDesiredPosition();
            }
        }

        
        private void readObject(ObjectInputStream anInputStream)
                throws ClassNotFoundException, IOException {
            // always perform the default de-serialization first
            anInputStream.defaultReadObject();
            // Build links
            for (int i = 1; i < PSQUARE_CONSTANT; i++) {
                markerArray[i].previous(markerArray[i - 1])
                        .next(markerArray[i + 1]).index(i);
            }
            markerArray[0].previous(markerArray[0]).next(markerArray[1])
                    .index(0);
            markerArray[5].previous(markerArray[4]).next(markerArray[5])
                    .index(5);
        }

        
        public double height(final int markerIndex) {
            if (markerIndex >= markerArray.length || markerIndex <= 0) {
                throw new OutOfRangeException(markerIndex, 1,
                        markerArray.length);
            }
            return markerArray[markerIndex].markerHeight;
        }

        
        @Override
        public Object clone() {
            return new Markers(new Marker[] { new Marker(),
                    (Marker) markerArray[1].clone(),
                    (Marker) markerArray[2].clone(),
                    (Marker) markerArray[3].clone(),
                    (Marker) markerArray[4].clone(),
                    (Marker) markerArray[5].clone() });

        }

        
        @Override
        public String toString() {
            return String.format("m1=[%s],m2=[%s],m3=[%s],m4=[%s],m5=[%s]",
                    markerArray[1].toString(), markerArray[2].toString(),
                    markerArray[3].toString(), markerArray[4].toString(),
                    markerArray[5].toString());
        }

    }

    
    private static class Marker implements Serializable, Cloneable {

        
        private static final long serialVersionUID = -3575879478288538431L;

        
        private int index;

        
        private double intMarkerPosition;

        
        private double desiredMarkerPosition;

        
        private double markerHeight;

        
        private double desiredMarkerIncrement;

        
        private transient Marker next;

        
        private transient Marker previous;

        
        private final UnivariateInterpolator nonLinear =
                new NevilleInterpolator();

        
        private transient UnivariateInterpolator linear =
                new LinearInterpolator();

        
        private Marker() {
            this.next = this.previous = this;
        }

        
        private Marker(double heightOfMarker, double makerPositionDesired,
                double markerPositionIncrement, double markerPositionNumber) {
            this();
            this.markerHeight = heightOfMarker;
            this.desiredMarkerPosition = makerPositionDesired;
            this.desiredMarkerIncrement = markerPositionIncrement;
            this.intMarkerPosition = markerPositionNumber;
        }

        
        private Marker previous(final Marker previousMarker) {
            MathUtils.checkNotNull(previousMarker);
            this.previous = previousMarker;
            return this;
        }

        
        private Marker next(final Marker nextMarker) {
            MathUtils.checkNotNull(nextMarker);
            this.next = nextMarker;
            return this;
        }

        
        private Marker index(final int indexOfMarker) {
            this.index = indexOfMarker;
            return this;
        }

        
        private void updateDesiredPosition() {
            desiredMarkerPosition += desiredMarkerIncrement;
        }

        
        private void incrementPosition(final int d) {
            intMarkerPosition += d;
        }

        
        private double difference() {
            return desiredMarkerPosition - intMarkerPosition;
        }

        
        private double estimate() {
            final double di = difference();
            final boolean isNextHigher =
                    next.intMarkerPosition - intMarkerPosition > 1;
            final boolean isPreviousLower =
                    previous.intMarkerPosition - intMarkerPosition < -1;

            if (di >= 1 && isNextHigher || di <= -1 && isPreviousLower) {
                final int d = di >= 0 ? 1 : -1;
                final double[] xval =
                        new double[] { previous.intMarkerPosition,
                                intMarkerPosition, next.intMarkerPosition };
                final double[] yval =
                        new double[] { previous.markerHeight, markerHeight,
                                next.markerHeight };
                final double xD = intMarkerPosition + d;

                UnivariateFunction univariateFunction =
                        nonLinear.interpolate(xval, yval);
                markerHeight = univariateFunction.value(xD);

                // If parabolic estimate is bad then turn linear
                if (isEstimateBad(yval, markerHeight)) {
                    int delta = xD - xval[1] > 0 ? 1 : -1;
                    final double[] xBad =
                            new double[] { xval[1], xval[1 + delta] };
                    final double[] yBad =
                            new double[] { yval[1], yval[1 + delta] };
                    MathArrays.sortInPlace(xBad, yBad);// since d can be +/- 1
                    univariateFunction = linear.interpolate(xBad, yBad);
                    markerHeight = univariateFunction.value(xD);
                }
                incrementPosition(d);
            }
            return markerHeight;
        }

        
        private boolean isEstimateBad(final double[] y, final double yD) {
            return yD <= y[0] || yD >= y[2];
        }

        
        @Override
        public boolean equals(Object o) {
            boolean result = false;
            if (this == o) {
                result = true;
            } else if (o != null && o instanceof Marker) {
                Marker that = (Marker) o;

                result = Double.compare(markerHeight, that.markerHeight) == 0;
                result =
                        result &&
                                Double.compare(intMarkerPosition,
                                        that.intMarkerPosition) == 0;
                result =
                        result &&
                                Double.compare(desiredMarkerPosition,
                                        that.desiredMarkerPosition) == 0;
                result =
                        result &&
                                Double.compare(desiredMarkerIncrement,
                                        that.desiredMarkerIncrement) == 0;

                result = result && next.index == that.next.index;
                result = result && previous.index == that.previous.index;
            }
            return result;
        }

        
        @Override
        public int hashCode() {
            return Arrays.hashCode(new double[] {markerHeight, intMarkerPosition,
                desiredMarkerIncrement, desiredMarkerPosition, previous.index, next.index});
        }

        
        private void readObject(ObjectInputStream anInstream)
                throws ClassNotFoundException, IOException {
            anInstream.defaultReadObject();
            previous=next=this;
            linear = new LinearInterpolator();
        }

        
        @Override
        public Object clone() {
            return new Marker(markerHeight, desiredMarkerPosition,
                    desiredMarkerIncrement, intMarkerPosition);
        }

        
        @Override
        public String toString() {
            return String.format(
                    "index=%.0f,n=%.0f,np=%.2f,q=%.2f,dn=%.2f,prev=%d,next=%d",
                    (double) index, Precision.round(intMarkerPosition, 0),
                    Precision.round(desiredMarkerPosition, 2),
                    Precision.round(markerHeight, 2),
                    Precision.round(desiredMarkerIncrement, 2), previous.index,
                    next.index);
        }
    }

    
    private static class FixedCapacityList<E> extends ArrayList<E> implements
            Serializable {
        
        private static final long serialVersionUID = 2283952083075725479L;
        
        private final int capacity;

        
        FixedCapacityList(final int fixedCapacity) {
            super(fixedCapacity);
            this.capacity = fixedCapacity;
        }

        
        @Override
        public boolean add(final E e) {
            return size() < capacity ? super.add(e) : false;
        }

        
        @Override
        public boolean addAll(Collection<? extends E> collection) {
            boolean isCollectionLess =
                    collection != null &&
                            collection.size() + size() <= capacity;
            return isCollectionLess ? super.addAll(collection) : false;
        }
    }

    
    public static PSquareMarkers newMarkers(final List<Double> initialFive,
            final double p) {
        return new Markers(initialFive, p);
    }

    
    protected interface PSquareMarkers extends Cloneable {
        
        double getPercentileValue();

        
        Object clone();

        
        double height(final int markerIndex);

        
        double processDataPoint(final double inputDataPoint);

        
        double estimate(final int index);
    }
}
