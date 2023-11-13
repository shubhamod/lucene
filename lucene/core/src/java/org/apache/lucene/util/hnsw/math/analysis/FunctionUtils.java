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

package org.apache.lucene.util.hnsw.math.analysis;

import org.apache.lucene.util.hnsw.math.analysis.differentiation.DerivativeStructure;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.MultivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.MultivariateDifferentiableVectorFunction;
import org.apache.lucene.util.hnsw.math.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.lucene.util.hnsw.math.analysis.function.Identity;
import org.apache.lucene.util.hnsw.math.exception.DimensionMismatchException;
import org.apache.lucene.util.hnsw.math.exception.NotStrictlyPositiveException;
import org.apache.lucene.util.hnsw.math.exception.NumberIsTooLargeException;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;


public class FunctionUtils {
    
    private FunctionUtils() {}

    
    public static UnivariateFunction compose(final UnivariateFunction ... f) {
        return new UnivariateFunction() {
            
            public double value(double x) {
                double r = x;
                for (int i = f.length - 1; i >= 0; i--) {
                    r = f[i].value(r);
                }
                return r;
            }
        };
    }

    
    public static UnivariateDifferentiableFunction compose(final UnivariateDifferentiableFunction ... f) {
        return new UnivariateDifferentiableFunction() {

            
            public double value(final double t) {
                double r = t;
                for (int i = f.length - 1; i >= 0; i--) {
                    r = f[i].value(r);
                }
                return r;
            }

            
            public DerivativeStructure value(final DerivativeStructure t) {
                DerivativeStructure r = t;
                for (int i = f.length - 1; i >= 0; i--) {
                    r = f[i].value(r);
                }
                return r;
            }

        };
    }

    
    @Deprecated
    public static DifferentiableUnivariateFunction compose(final DifferentiableUnivariateFunction ... f) {
        return new DifferentiableUnivariateFunction() {
            
            public double value(double x) {
                double r = x;
                for (int i = f.length - 1; i >= 0; i--) {
                    r = f[i].value(r);
                }
                return r;
            }

            
            public UnivariateFunction derivative() {
                return new UnivariateFunction() {
                    
                    public double value(double x) {
                        double p = 1;
                        double r = x;
                        for (int i = f.length - 1; i >= 0; i--) {
                            p *= f[i].derivative().value(r);
                            r = f[i].value(r);
                        }
                        return p;
                    }
                };
            }
        };
    }

    
    public static UnivariateFunction add(final UnivariateFunction ... f) {
        return new UnivariateFunction() {
            
            public double value(double x) {
                double r = f[0].value(x);
                for (int i = 1; i < f.length; i++) {
                    r += f[i].value(x);
                }
                return r;
            }
        };
    }

    
    public static UnivariateDifferentiableFunction add(final UnivariateDifferentiableFunction ... f) {
        return new UnivariateDifferentiableFunction() {

            
            public double value(final double t) {
                double r = f[0].value(t);
                for (int i = 1; i < f.length; i++) {
                    r += f[i].value(t);
                }
                return r;
            }

            
            public DerivativeStructure value(final DerivativeStructure t)
                throws DimensionMismatchException {
                DerivativeStructure r = f[0].value(t);
                for (int i = 1; i < f.length; i++) {
                    r = r.add(f[i].value(t));
                }
                return r;
            }

        };
    }

    
    @Deprecated
    public static DifferentiableUnivariateFunction add(final DifferentiableUnivariateFunction ... f) {
        return new DifferentiableUnivariateFunction() {
            
            public double value(double x) {
                double r = f[0].value(x);
                for (int i = 1; i < f.length; i++) {
                    r += f[i].value(x);
                }
                return r;
            }

            
            public UnivariateFunction derivative() {
                return new UnivariateFunction() {
                    
                    public double value(double x) {
                        double r = f[0].derivative().value(x);
                        for (int i = 1; i < f.length; i++) {
                            r += f[i].derivative().value(x);
                        }
                        return r;
                    }
                };
            }
        };
    }

    
    public static UnivariateFunction multiply(final UnivariateFunction ... f) {
        return new UnivariateFunction() {
            
            public double value(double x) {
                double r = f[0].value(x);
                for (int i = 1; i < f.length; i++) {
                    r *= f[i].value(x);
                }
                return r;
            }
        };
    }

    
    public static UnivariateDifferentiableFunction multiply(final UnivariateDifferentiableFunction ... f) {
        return new UnivariateDifferentiableFunction() {

            
            public double value(final double t) {
                double r = f[0].value(t);
                for (int i = 1; i < f.length; i++) {
                    r  *= f[i].value(t);
                }
                return r;
            }

            
            public DerivativeStructure value(final DerivativeStructure t) {
                DerivativeStructure r = f[0].value(t);
                for (int i = 1; i < f.length; i++) {
                    r = r.multiply(f[i].value(t));
                }
                return r;
            }

        };
    }

    
    @Deprecated
    public static DifferentiableUnivariateFunction multiply(final DifferentiableUnivariateFunction ... f) {
        return new DifferentiableUnivariateFunction() {
            
            public double value(double x) {
                double r = f[0].value(x);
                for (int i = 1; i < f.length; i++) {
                    r *= f[i].value(x);
                }
                return r;
            }

            
            public UnivariateFunction derivative() {
                return new UnivariateFunction() {
                    
                    public double value(double x) {
                        double sum = 0;
                        for (int i = 0; i < f.length; i++) {
                            double prod = f[i].derivative().value(x);
                            for (int j = 0; j < f.length; j++) {
                                if (i != j) {
                                    prod *= f[j].value(x);
                                }
                            }
                            sum += prod;
                        }
                        return sum;
                    }
                };
            }
        };
    }

    
    public static UnivariateFunction combine(final BivariateFunction combiner,
                                             final UnivariateFunction f,
                                             final UnivariateFunction g) {
        return new UnivariateFunction() {
            
            public double value(double x) {
                return combiner.value(f.value(x), g.value(x));
            }
        };
    }

    
    public static MultivariateFunction collector(final BivariateFunction combiner,
                                                 final UnivariateFunction f,
                                                 final double initialValue) {
        return new MultivariateFunction() {
            
            public double value(double[] point) {
                double result = combiner.value(initialValue, f.value(point[0]));
                for (int i = 1; i < point.length; i++) {
                    result = combiner.value(result, f.value(point[i]));
                }
                return result;
            }
        };
    }

    
    public static MultivariateFunction collector(final BivariateFunction combiner,
                                                 final double initialValue) {
        return collector(combiner, new Identity(), initialValue);
    }

    
    public static UnivariateFunction fix1stArgument(final BivariateFunction f,
                                                    final double fixed) {
        return new UnivariateFunction() {
            
            public double value(double x) {
                return f.value(fixed, x);
            }
        };
    }
    
    public static UnivariateFunction fix2ndArgument(final BivariateFunction f,
                                                    final double fixed) {
        return new UnivariateFunction() {
            
            public double value(double x) {
                return f.value(x, fixed);
            }
        };
    }

    
    public static double[] sample(UnivariateFunction f, double min, double max, int n)
       throws NumberIsTooLargeException, NotStrictlyPositiveException {

        if (n <= 0) {
            throw new NotStrictlyPositiveException(
                    LocalizedFormats.NOT_POSITIVE_NUMBER_OF_SAMPLES,
                    Integer.valueOf(n));
        }
        if (min >= max) {
            throw new NumberIsTooLargeException(min, max, false);
        }

        final double[] s = new double[n];
        final double h = (max - min) / n;
        for (int i = 0; i < n; i++) {
            s[i] = f.value(min + i * h);
        }
        return s;
    }

    
    @Deprecated
    public static DifferentiableUnivariateFunction toDifferentiableUnivariateFunction(final UnivariateDifferentiableFunction f) {
        return new DifferentiableUnivariateFunction() {

            
            public double value(final double x) {
                return f.value(x);
            }

            
            public UnivariateFunction derivative() {
                return new UnivariateFunction() {
                    
                    public double value(final double x) {
                        return f.value(new DerivativeStructure(1, 1, 0, x)).getPartialDerivative(1);
                    }
                };
            }

        };
    }

    
    @Deprecated
    public static UnivariateDifferentiableFunction toUnivariateDifferential(final DifferentiableUnivariateFunction f) {
        return new UnivariateDifferentiableFunction() {

            
            public double value(final double x) {
                return f.value(x);
            }

            
            public DerivativeStructure value(final DerivativeStructure t)
                throws NumberIsTooLargeException {
                switch (t.getOrder()) {
                    case 0 :
                        return new DerivativeStructure(t.getFreeParameters(), 0, f.value(t.getValue()));
                    case 1 : {
                        final int parameters = t.getFreeParameters();
                        final double[] derivatives = new double[parameters + 1];
                        derivatives[0] = f.value(t.getValue());
                        final double fPrime = f.derivative().value(t.getValue());
                        int[] orders = new int[parameters];
                        for (int i = 0; i < parameters; ++i) {
                            orders[i] = 1;
                            derivatives[i + 1] = fPrime * t.getPartialDerivative(orders);
                            orders[i] = 0;
                        }
                        return new DerivativeStructure(parameters, 1, derivatives);
                    }
                    default :
                        throw new NumberIsTooLargeException(t.getOrder(), 1, true);
                }
            }

        };
    }

    
    @Deprecated
    public static DifferentiableMultivariateFunction toDifferentiableMultivariateFunction(final MultivariateDifferentiableFunction f) {
        return new DifferentiableMultivariateFunction() {

            
            public double value(final double[] x) {
                return f.value(x);
            }

            
            public MultivariateFunction partialDerivative(final int k) {
                return new MultivariateFunction() {
                    
                    public double value(final double[] x) {

                        final int n = x.length;

                        // delegate computation to underlying function
                        final DerivativeStructure[] dsX = new DerivativeStructure[n];
                        for (int i = 0; i < n; ++i) {
                            if (i == k) {
                                dsX[i] = new DerivativeStructure(1, 1, 0, x[i]);
                            } else {
                                dsX[i] = new DerivativeStructure(1, 1, x[i]);
                            }
                        }
                        final DerivativeStructure y = f.value(dsX);

                        // extract partial derivative
                        return y.getPartialDerivative(1);

                    }
                };
            }

            
            public MultivariateVectorFunction gradient() {
                return new MultivariateVectorFunction() {
                    
                    public double[] value(final double[] x) {

                        final int n = x.length;

                        // delegate computation to underlying function
                        final DerivativeStructure[] dsX = new DerivativeStructure[n];
                        for (int i = 0; i < n; ++i) {
                            dsX[i] = new DerivativeStructure(n, 1, i, x[i]);
                        }
                        final DerivativeStructure y = f.value(dsX);

                        // extract gradient
                        final double[] gradient = new double[n];
                        final int[] orders = new int[n];
                        for (int i = 0; i < n; ++i) {
                            orders[i]   = 1;
                            gradient[i] = y.getPartialDerivative(orders);
                            orders[i]   = 0;
                        }

                        return gradient;

                    }
                };
            }

        };
    }

    
    @Deprecated
    public static MultivariateDifferentiableFunction toMultivariateDifferentiableFunction(final DifferentiableMultivariateFunction f) {
        return new MultivariateDifferentiableFunction() {

            
            public double value(final double[] x) {
                return f.value(x);
            }

            
            public DerivativeStructure value(final DerivativeStructure[] t)
                throws DimensionMismatchException, NumberIsTooLargeException {

                // check parameters and orders limits
                final int parameters = t[0].getFreeParameters();
                final int order      = t[0].getOrder();
                final int n          = t.length;
                if (order > 1) {
                    throw new NumberIsTooLargeException(order, 1, true);
                }

                // check all elements in the array are consistent
                for (int i = 0; i < n; ++i) {
                    if (t[i].getFreeParameters() != parameters) {
                        throw new DimensionMismatchException(t[i].getFreeParameters(), parameters);
                    }

                    if (t[i].getOrder() != order) {
                        throw new DimensionMismatchException(t[i].getOrder(), order);
                    }
                }

                // delegate computation to underlying function
                final double[] point = new double[n];
                for (int i = 0; i < n; ++i) {
                    point[i] = t[i].getValue();
                }
                final double value      = f.value(point);
                final double[] gradient = f.gradient().value(point);

                // merge value and gradient into one DerivativeStructure
                final double[] derivatives = new double[parameters + 1];
                derivatives[0] = value;
                final int[] orders = new int[parameters];
                for (int i = 0; i < parameters; ++i) {
                    orders[i] = 1;
                    for (int j = 0; j < n; ++j) {
                        derivatives[i + 1] += gradient[j] * t[j].getPartialDerivative(orders);
                    }
                    orders[i] = 0;
                }

                return new DerivativeStructure(parameters, order, derivatives);

            }

        };
    }

    
    @Deprecated
    public static DifferentiableMultivariateVectorFunction toDifferentiableMultivariateVectorFunction(final MultivariateDifferentiableVectorFunction f) {
        return new DifferentiableMultivariateVectorFunction() {

            
            public double[] value(final double[] x) {
                return f.value(x);
            }

            
            public MultivariateMatrixFunction jacobian() {
                return new MultivariateMatrixFunction() {
                    
                    public double[][] value(final double[] x) {

                        final int n = x.length;

                        // delegate computation to underlying function
                        final DerivativeStructure[] dsX = new DerivativeStructure[n];
                        for (int i = 0; i < n; ++i) {
                            dsX[i] = new DerivativeStructure(n, 1, i, x[i]);
                        }
                        final DerivativeStructure[] y = f.value(dsX);

                        // extract Jacobian
                        final double[][] jacobian = new double[y.length][n];
                        final int[] orders = new int[n];
                        for (int i = 0; i < y.length; ++i) {
                            for (int j = 0; j < n; ++j) {
                                orders[j]      = 1;
                                jacobian[i][j] = y[i].getPartialDerivative(orders);
                                orders[j]      = 0;
                            }
                        }

                        return jacobian;

                    }
                };
            }

        };
    }

    
    @Deprecated
    public static MultivariateDifferentiableVectorFunction toMultivariateDifferentiableVectorFunction(final DifferentiableMultivariateVectorFunction f) {
        return new MultivariateDifferentiableVectorFunction() {

            
            public double[] value(final double[] x) {
                return f.value(x);
            }

            
            public DerivativeStructure[] value(final DerivativeStructure[] t)
                throws DimensionMismatchException, NumberIsTooLargeException {

                // check parameters and orders limits
                final int parameters = t[0].getFreeParameters();
                final int order      = t[0].getOrder();
                final int n          = t.length;
                if (order > 1) {
                    throw new NumberIsTooLargeException(order, 1, true);
                }

                // check all elements in the array are consistent
                for (int i = 0; i < n; ++i) {
                    if (t[i].getFreeParameters() != parameters) {
                        throw new DimensionMismatchException(t[i].getFreeParameters(), parameters);
                    }

                    if (t[i].getOrder() != order) {
                        throw new DimensionMismatchException(t[i].getOrder(), order);
                    }
                }

                // delegate computation to underlying function
                final double[] point = new double[n];
                for (int i = 0; i < n; ++i) {
                    point[i] = t[i].getValue();
                }
                final double[] value      = f.value(point);
                final double[][] jacobian = f.jacobian().value(point);

                // merge value and Jacobian into a DerivativeStructure array
                final DerivativeStructure[] merged = new DerivativeStructure[value.length];
                for (int k = 0; k < merged.length; ++k) {
                    final double[] derivatives = new double[parameters + 1];
                    derivatives[0] = value[k];
                    final int[] orders = new int[parameters];
                    for (int i = 0; i < parameters; ++i) {
                        orders[i] = 1;
                        for (int j = 0; j < n; ++j) {
                            derivatives[i + 1] += jacobian[k][j] * t[j].getPartialDerivative(orders);
                        }
                        orders[i] = 0;
                    }
                    merged[k] = new DerivativeStructure(parameters, order, derivatives);
                }

                return merged;

            }

        };
    }

}
