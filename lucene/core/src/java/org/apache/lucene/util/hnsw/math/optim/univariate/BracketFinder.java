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
package org.apache.lucene.util.hnsw.math.optim.univariate;

import org.apache.lucene.util.hnsw.math.util.FastMath;
import org.apache.lucene.util.hnsw.math.util.IntegerSequence;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.TooManyEvaluationsException;
import org.apache.lucene.util.hnsw.math.exception.MaxCountExceededException;
import org.apache.lucene.util.hnsw.math.analysis.UnivariateFunction;
import org.apache.lucene.util.hnsw.math.optim.nonlinear.scalar.GoalType;


public class BracketFinder {
    
    private static final double EPS_MIN = 1e-21;
    
    private static final double GOLD = 1.618034;
    
    private final double growLimit;
    
    private IntegerSequence.Incrementor evaluations;
    
    private double lo;
    
    private double hi;
    
    private double mid;
    
    private double fLo;
    
    private double fHi;
    
    private double fMid;

    
    public BracketFinder() {
        this(100, 500);
    }

    
    public BracketFinder(double growLimit,
                         int maxEvaluations) {
        if (growLimit <= 0) {
            throw new NotStrictlyPositiveException(growLimit);
        }
        if (maxEvaluations <= 0) {
            throw new NotStrictlyPositiveException(maxEvaluations);
        }

        this.growLimit = growLimit;
        evaluations = IntegerSequence.Incrementor.create().withMaximalCount(maxEvaluations);
    }

    
    public void search(UnivariateFunction func,
                       GoalType goal,
                       double xA,
                       double xB) {
        evaluations = evaluations.withStart(0);
        final boolean isMinim = goal == GoalType.MINIMIZE;

        double fA = eval(func, xA);
        double fB = eval(func, xB);
        if (isMinim ?
            fA < fB :
            fA > fB) {

            double tmp = xA;
            xA = xB;
            xB = tmp;

            tmp = fA;
            fA = fB;
            fB = tmp;
        }

        double xC = xB + GOLD * (xB - xA);
        double fC = eval(func, xC);

        while (isMinim ? fC < fB : fC > fB) {
            double tmp1 = (xB - xA) * (fB - fC);
            double tmp2 = (xB - xC) * (fB - fA);

            double val = tmp2 - tmp1;
            double denom = FastMath.abs(val) < EPS_MIN ? 2 * EPS_MIN : 2 * val;

            double w = xB - ((xB - xC) * tmp2 - (xB - xA) * tmp1) / denom;
            double wLim = xB + growLimit * (xC - xB);

            double fW;
            if ((w - xC) * (xB - w) > 0) {
                fW = eval(func, w);
                if (isMinim ?
                    fW < fC :
                    fW > fC) {
                    xA = xB;
                    xB = w;
                    fA = fB;
                    fB = fW;
                    break;
                } else if (isMinim ?
                           fW > fB :
                           fW < fB) {
                    xC = w;
                    fC = fW;
                    break;
                }
                w = xC + GOLD * (xC - xB);
                fW = eval(func, w);
            } else if ((w - wLim) * (wLim - xC) >= 0) {
                w = wLim;
                fW = eval(func, w);
            } else if ((w - wLim) * (xC - w) > 0) {
                fW = eval(func, w);
                if (isMinim ?
                    fW < fC :
                    fW > fC) {
                    xB = xC;
                    xC = w;
                    w = xC + GOLD * (xC - xB);
                    fB = fC;
                    fC =fW;
                    fW = eval(func, w);
                }
            } else {
                w = xC + GOLD * (xC - xB);
                fW = eval(func, w);
            }

            xA = xB;
            fA = fB;
            xB = xC;
            fB = fC;
            xC = w;
            fC = fW;
        }

        lo = xA;
        fLo = fA;
        mid = xB;
        fMid = fB;
        hi = xC;
        fHi = fC;

        if (lo > hi) {
            double tmp = lo;
            lo = hi;
            hi = tmp;

            tmp = fLo;
            fLo = fHi;
            fHi = tmp;
        }
    }

    
    public int getMaxEvaluations() {
        return evaluations.getMaximalCount();
    }

    
    public int getEvaluations() {
        return evaluations.getCount();
    }

    
    public double getLo() {
        return lo;
    }

    
    public double getFLo() {
        return fLo;
    }

    
    public double getHi() {
        return hi;
    }

    
    public double getFHi() {
        return fHi;
    }

    
    public double getMid() {
        return mid;
    }

    
    public double getFMid() {
        return fMid;
    }

    
    private double eval(UnivariateFunction f, double x) {
        try {
            evaluations.increment();
        } catch (MaxCountExceededException e) {
            throw new TooManyEvaluationsException(e.getMax());
        }
        return f.value(x);
    }
}
