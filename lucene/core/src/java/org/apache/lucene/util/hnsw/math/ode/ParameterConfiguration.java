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
package org.apache.lucene.util.hnsw.math.ode;

import java.io.Serializable;


class ParameterConfiguration implements Serializable {

    
    private static final long serialVersionUID = 2247518849090889379L;

    
    private String parameterName;

    
    private double hP;

    
    ParameterConfiguration(final String parameterName, final double hP) {
        this.parameterName = parameterName;
        this.hP = hP;
    }

    
    public String getParameterName() {
        return parameterName;
    }

    
    public double getHP() {
        return hP;
    }

    
    public void setHP(final double hParam) {
        this.hP = hParam;
    }

}
