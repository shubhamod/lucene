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
package org.apache.lucene.util.hnsw.math.exception;

import org.apache.lucene.util.hnsw.math.exception.util.Localizable;
import org.apache.lucene.util.hnsw.math.exception.util.LocalizedFormats;
import org.apache.lucene.util.hnsw.math.exception.util.ExceptionContext;
import org.apache.lucene.util.hnsw.math.exception.util.ExceptionContextProvider;


public class MathArithmeticException extends ArithmeticException
    implements ExceptionContextProvider {
    
    private static final long serialVersionUID = -6024911025449780478L;
    
    private final ExceptionContext context;

    
    public MathArithmeticException() {
        context = new ExceptionContext(this);
        context.addMessage(LocalizedFormats.ARITHMETIC_EXCEPTION);
    }

    
    public MathArithmeticException(Localizable pattern,
                                   Object ... args) {
        context = new ExceptionContext(this);
        context.addMessage(pattern, args);
    }

    
    public ExceptionContext getContext() {
        return context;
    }

    
    @Override
    public String getMessage() {
        return context.getMessage();
    }

    
    @Override
    public String getLocalizedMessage() {
        return context.getLocalizedMessage();
    }
}
