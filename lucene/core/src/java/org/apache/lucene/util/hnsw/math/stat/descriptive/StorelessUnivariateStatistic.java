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
package org.apache.lucene.util.hnsw.math.stat.descriptive;

import org.apache.lucene.util.hnsw.math.exception.MathIllegalArgumentException;


public interface StorelessUnivariateStatistic extends UnivariateStatistic {

    
    void increment(double d);

    
    void incrementAll(double[] values) throws MathIllegalArgumentException;

    
    void incrementAll(double[] values, int start, int length) throws MathIllegalArgumentException;

    
    double getResult();

    
    long getN();

    
    void clear();

    
    StorelessUnivariateStatistic copy();

}
