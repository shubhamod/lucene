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
package org.apache.lucene.util.hnsw.math.random;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.NoSuchElementException;
import java.util.StringTokenizer;

import org.apache.lucene.util.hnsw.math.exception.MathInternalError;
import org.apache.lucene.util.hnsw.math.exception.MathParseException;
import org.apache.lucene.util.hnsw.math.exception.NotPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.OutOfRangeException;
import org.apache.lucene.util.hnsw.math.util.FastMath;


public class SobolSequenceGenerator implements RandomVectorGenerator {

    
    private static final int BITS = 52;

    
    private static final double SCALE = FastMath.pow(2, BITS);

    
    private static final int MAX_DIMENSION = 1000;

    
    private static final String RESOURCE_NAME = "/assets/org/apache/commons/math3/random/new-joe-kuo-6.1000";

    
    private static final String FILE_CHARSET = "US-ASCII";

    
    private final int dimension;

    
    private int count = 0;

    
    private final long[][] direction;

    
    private final long[] x;

    
    public SobolSequenceGenerator(final int dimension) throws OutOfRangeException {
        if (dimension < 1 || dimension > MAX_DIMENSION) {
            throw new OutOfRangeException(dimension, 1, MAX_DIMENSION);
        }

        // initialize the other dimensions with direction numbers from a resource
        final InputStream is = getClass().getResourceAsStream(RESOURCE_NAME);
        if (is == null) {
            throw new MathInternalError();
        }

        this.dimension = dimension;

        // init data structures
        direction = new long[dimension][BITS + 1];
        x = new long[dimension];

        try {
            initFromStream(is);
        } catch (IOException e) {
            // the internal resource file could not be read -> should not happen
            throw new MathInternalError();
        } catch (MathParseException e) {
            // the internal resource file could not be parsed -> should not happen
            throw new MathInternalError();
        } finally {
            try {
                is.close();
            } catch (IOException e) { // NOPMD
                // ignore
            }
        }
    }

    
    public SobolSequenceGenerator(final int dimension, final InputStream is)
            throws NotStrictlyPositiveException, MathParseException, IOException {

        if (dimension < 1) {
            throw new NotStrictlyPositiveException(dimension);
        }

        this.dimension = dimension;

        // init data structures
        direction = new long[dimension][BITS + 1];
        x = new long[dimension];

        // initialize the other dimensions with direction numbers from the stream
        int lastDimension = initFromStream(is);
        if (lastDimension < dimension) {
            throw new OutOfRangeException(dimension, 1, lastDimension);
        }
    }

    
    private int initFromStream(final InputStream is) throws MathParseException, IOException {

        // special case: dimension 1 -> use unit initialization
        for (int i = 1; i <= BITS; i++) {
            direction[0][i] = 1l << (BITS - i);
        }

        final Charset charset = Charset.forName(FILE_CHARSET);
        final BufferedReader reader = new BufferedReader(new InputStreamReader(is, charset));
        int dim = -1;

        try {
            // ignore first line
            reader.readLine();

            int lineNumber = 2;
            int index = 1;
            String line = null;
            while ( (line = reader.readLine()) != null) {
                StringTokenizer st = new StringTokenizer(line, " ");
                try {
                    dim = Integer.parseInt(st.nextToken());
                    if (dim >= 2 && dim <= dimension) { // we have found the right dimension
                        final int s = Integer.parseInt(st.nextToken());
                        final int a = Integer.parseInt(st.nextToken());
                        final int[] m = new int[s + 1];
                        for (int i = 1; i <= s; i++) {
                            m[i] = Integer.parseInt(st.nextToken());
                        }
                        initDirectionVector(index++, a, m);
                    }

                    if (dim > dimension) {
                        return dim;
                    }
                } catch (NoSuchElementException e) {
                    throw new MathParseException(line, lineNumber);
                } catch (NumberFormatException e) {
                    throw new MathParseException(line, lineNumber);
                }
                lineNumber++;
            }
        } finally {
            reader.close();
        }

        return dim;
    }

    
    private void initDirectionVector(final int d, final int a, final int[] m) {
        final int s = m.length - 1;
        for (int i = 1; i <= s; i++) {
            direction[d][i] = ((long) m[i]) << (BITS - i);
        }
        for (int i = s + 1; i <= BITS; i++) {
            direction[d][i] = direction[d][i - s] ^ (direction[d][i - s] >> s);
            for (int k = 1; k <= s - 1; k++) {
                direction[d][i] ^= ((a >> (s - 1 - k)) & 1) * direction[d][i - k];
            }
        }
    }

    
    public double[] nextVector() {
        final double[] v = new double[dimension];
        if (count == 0) {
            count++;
            return v;
        }

        // find the index c of the rightmost 0
        int c = 1;
        int value = count - 1;
        while ((value & 1) == 1) {
            value >>= 1;
            c++;
        }

        for (int i = 0; i < dimension; i++) {
            x[i] ^= direction[i][c];
            v[i] = (double) x[i] / SCALE;
        }
        count++;
        return v;
    }

    
    public double[] skipTo(final int index) throws NotPositiveException {
        if (index == 0) {
            // reset x vector
            Arrays.fill(x, 0);
        } else {
            final int i = index - 1;
            final long grayCode = i ^ (i >> 1); // compute the gray code of i = i XOR floor(i / 2)
            for (int j = 0; j < dimension; j++) {
                long result = 0;
                for (int k = 1; k <= BITS; k++) {
                    final long shift = grayCode >> (k - 1);
                    if (shift == 0) {
                        // stop, as all remaining bits will be zero
                        break;
                    }
                    // the k-th bit of i
                    final long ik = shift & 1;
                    result ^= ik * direction[j][k];
                }
                x[j] = result;
            }
        }
        count = index;
        return nextVector();
    }

    
    public int getNextIndex() {
        return count;
    }

}
