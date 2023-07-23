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

import java.util.ArrayList;
import java.util.Collection;



public abstract class AbstractParameterizable implements Parameterizable {

   
    private final Collection<String> parametersNames;

    
    protected AbstractParameterizable(final String ... names) {
        parametersNames = new ArrayList<String>();
        for (final String name : names) {
            parametersNames.add(name);
        }
    }

    
    protected AbstractParameterizable(final Collection<String> names) {
        parametersNames = new ArrayList<String>();
        parametersNames.addAll(names);
    }

    
    public Collection<String> getParametersNames() {
        return parametersNames;
    }

    
    public boolean isSupported(final String name) {
        for (final String supportedName : parametersNames) {
            if (supportedName.equals(name)) {
                return true;
            }
        }
        return false;
    }

    
    public void complainIfNotSupported(final String name)
        throws UnknownParameterException {
        if (!isSupported(name)) {
            throw new UnknownParameterException(name);
        }
    }

}
