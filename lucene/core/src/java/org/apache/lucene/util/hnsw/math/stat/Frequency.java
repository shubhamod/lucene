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
package org.apache.lucene.util.hnsw.math.stat;

import java.io.Serializable;
import java.text.NumberFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;
import org.apache.lucene.util.hnsw.math.exception.NullArgumentException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.util.MathUtils;


public class Frequency implements Serializable {

    
    private static final long serialVersionUID = -3845586908418844111L;

    
    private final SortedMap<Comparable<?>, Long> freqTable;

    
    public Frequency() {
        freqTable = new TreeMap<Comparable<?>, Long>();
    }

    
    @SuppressWarnings("unchecked") // TODO is the cast OK?
    public Frequency(Comparator<?> comparator) {
        freqTable = new TreeMap<Comparable<?>, Long>((Comparator<? super Comparable<?>>) comparator);
    }

    
    @Override
    public String toString() {
        NumberFormat nf = NumberFormat.getPercentInstance();
        StringBuilder outBuffer = new StringBuilder();
        outBuffer.append("Value \t Freq. \t Pct. \t Cum Pct. \n");
        Iterator<Comparable<?>> iter = freqTable.keySet().iterator();
        while (iter.hasNext()) {
            Comparable<?> value = iter.next();
            outBuffer.append(value);
            outBuffer.append('\t');
            outBuffer.append(getCount(value));
            outBuffer.append('\t');
            outBuffer.append(nf.format(getPct(value)));
            outBuffer.append('\t');
            outBuffer.append(nf.format(getCumPct(value)));
            outBuffer.append('\n');
        }
        return outBuffer.toString();
    }

    
    public void addValue(Comparable<?> v) throws MathIllegalArgumentException {
        incrementValue(v, 1);
    }

    
    public void addValue(int v) throws MathIllegalArgumentException {
        addValue(Long.valueOf(v));
    }

    
    public void addValue(long v) throws MathIllegalArgumentException {
        addValue(Long.valueOf(v));
    }

    
    public void addValue(char v) throws MathIllegalArgumentException {
        addValue(Character.valueOf(v));
    }

    
    public void incrementValue(Comparable<?> v, long increment) throws MathIllegalArgumentException {
        Comparable<?> obj = v;
        if (v instanceof Integer) {
            obj = Long.valueOf(((Integer) v).longValue());
        }
        try {
            Long count = freqTable.get(obj);
            if (count == null) {
                freqTable.put(obj, Long.valueOf(increment));
            } else {
                freqTable.put(obj, Long.valueOf(count.longValue() + increment));
            }
        } catch (ClassCastException ex) {
            //TreeMap will throw ClassCastException if v is not comparable
            throw new MathIllegalArgumentException(
                  LocalizedFormats.INSTANCES_NOT_COMPARABLE_TO_EXISTING_VALUES,
                  v.getClass().getName());
        }
    }

    
    public void incrementValue(int v, long increment) throws MathIllegalArgumentException {
        incrementValue(Long.valueOf(v), increment);
    }

    
    public void incrementValue(long v, long increment) throws MathIllegalArgumentException {
        incrementValue(Long.valueOf(v), increment);
    }

    
    public void incrementValue(char v, long increment) throws MathIllegalArgumentException {
        incrementValue(Character.valueOf(v), increment);
    }

    
    public void clear() {
        freqTable.clear();
    }

    
    public Iterator<Comparable<?>> valuesIterator() {
        return freqTable.keySet().iterator();
    }

    
    public Iterator<Entry<Comparable<?>, Long>> entrySetIterator() {
        return freqTable.entrySet().iterator();
    }

    //-------------------------------------------------------------------------

    
    public long getSumFreq() {
        long result = 0;
        Iterator<Long> iterator = freqTable.values().iterator();
        while (iterator.hasNext())  {
            result += iterator.next().longValue();
        }
        return result;
    }

    
    public long getCount(Comparable<?> v) {
        if (v instanceof Integer) {
            return getCount(((Integer) v).longValue());
        }
        long result = 0;
        try {
            Long count =  freqTable.get(v);
            if (count != null) {
                result = count.longValue();
            }
        } catch (ClassCastException ex) { // NOPMD
            // ignore and return 0 -- ClassCastException will be thrown if value is not comparable
        }
        return result;
    }

    
    public long getCount(int v) {
        return getCount(Long.valueOf(v));
    }

    
    public long getCount(long v) {
        return getCount(Long.valueOf(v));
    }

    
    public long getCount(char v) {
        return getCount(Character.valueOf(v));
    }

    
    public int getUniqueCount(){
        return freqTable.keySet().size();
    }

    
    public double getPct(Comparable<?> v) {
        final long sumFreq = getSumFreq();
        if (sumFreq == 0) {
            return Double.NaN;
        }
        return (double) getCount(v) / (double) sumFreq;
    }

    
    public double getPct(int v) {
        return getPct(Long.valueOf(v));
    }

    
    public double getPct(long v) {
        return getPct(Long.valueOf(v));
    }

    
    public double getPct(char v) {
        return getPct(Character.valueOf(v));
    }

    //-----------------------------------------------------------------------------------------

    
    @SuppressWarnings({ "rawtypes", "unchecked" })
    public long getCumFreq(Comparable<?> v) {
        if (getSumFreq() == 0) {
            return 0;
        }
        if (v instanceof Integer) {
            return getCumFreq(((Integer) v).longValue());
        }
        Comparator<Comparable<?>> c = (Comparator<Comparable<?>>) freqTable.comparator();
        if (c == null) {
            c = new NaturalComparator();
        }
        long result = 0;

        try {
            Long value = freqTable.get(v);
            if (value != null) {
                result = value.longValue();
            }
        } catch (ClassCastException ex) {
            return result;   // v is not comparable
        }

        if (c.compare(v, freqTable.firstKey()) < 0) {
            return 0;  // v is comparable, but less than first value
        }

        if (c.compare(v, freqTable.lastKey()) >= 0) {
            return getSumFreq();    // v is comparable, but greater than the last value
        }

        Iterator<Comparable<?>> values = valuesIterator();
        while (values.hasNext()) {
            Comparable<?> nextValue = values.next();
            if (c.compare(v, nextValue) > 0) {
                result += getCount(nextValue);
            } else {
                return result;
            }
        }
        return result;
    }

     
    public long getCumFreq(int v) {
        return getCumFreq(Long.valueOf(v));
    }

     
    public long getCumFreq(long v) {
        return getCumFreq(Long.valueOf(v));
    }

    
    public long getCumFreq(char v) {
        return getCumFreq(Character.valueOf(v));
    }

    //----------------------------------------------------------------------------------------------

    
    public double getCumPct(Comparable<?> v) {
        final long sumFreq = getSumFreq();
        if (sumFreq == 0) {
            return Double.NaN;
        }
        return (double) getCumFreq(v) / (double) sumFreq;
    }

    
    public double getCumPct(int v) {
        return getCumPct(Long.valueOf(v));
    }

    
    public double getCumPct(long v) {
        return getCumPct(Long.valueOf(v));
    }

    
    public double getCumPct(char v) {
        return getCumPct(Character.valueOf(v));
    }

    
    public List<Comparable<?>> getMode() {
        long mostPopular = 0; // frequencies are always positive

        // Get the max count first, so we avoid having to recreate the List each time
        for(Long l : freqTable.values()) {
            long frequency = l.longValue();
            if (frequency > mostPopular) {
                mostPopular = frequency;
            }
        }

        List<Comparable<?>> modeList = new ArrayList<Comparable<?>>();
        for (Entry<Comparable<?>, Long> ent : freqTable.entrySet()) {
            long frequency = ent.getValue().longValue();
            if (frequency == mostPopular) {
               modeList.add(ent.getKey());
            }
        }
        return modeList;
    }

    //----------------------------------------------------------------------------------------------

    
    public void merge(final Frequency other) throws NullArgumentException {
        MathUtils.checkNotNull(other, LocalizedFormats.NULL_NOT_ALLOWED);

        final Iterator<Entry<Comparable<?>, Long>> iter = other.entrySetIterator();
        while (iter.hasNext()) {
            final Entry<Comparable<?>, Long> entry = iter.next();
            incrementValue(entry.getKey(), entry.getValue().longValue());
        }
    }

    
    public void merge(final Collection<Frequency> others) throws NullArgumentException {
        MathUtils.checkNotNull(others, LocalizedFormats.NULL_NOT_ALLOWED);

        for (final Frequency freq : others) {
            merge(freq);
        }
    }

    //----------------------------------------------------------------------------------------------

    
    private static class NaturalComparator<T extends Comparable<T>> implements Comparator<Comparable<T>>, Serializable {

        
        private static final long serialVersionUID = -3852193713161395148L;

        
        @SuppressWarnings("unchecked") // cast to (T) may throw ClassCastException, see Javadoc
        public int compare(Comparable<T> o1, Comparable<T> o2) {
            return o1.compareTo((T) o2);
        }
    }

    
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result +
                 ((freqTable == null) ? 0 : freqTable.hashCode());
        return result;
    }

    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (!(obj instanceof Frequency)) {
            return false;
        }
        Frequency other = (Frequency) obj;
        if (freqTable == null) {
            if (other.freqTable != null) {
                return false;
            }
        } else if (!freqTable.equals(other.freqTable)) {
            return false;
        }
        return true;
    }

}
