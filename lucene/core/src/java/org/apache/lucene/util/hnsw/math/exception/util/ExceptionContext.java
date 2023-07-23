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
package org.apache.lucene.util.hnsw.math.exception.util;

import java.util.List;
import java.util.ArrayList;
import java.util.Set;
import java.util.Map;
import java.io.IOException;
import java.io.Serializable;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.util.HashMap;
import java.text.MessageFormat;
import java.util.Locale;


public class ExceptionContext implements Serializable {
    
    private static final long serialVersionUID = -6024911025449780478L;
    
    private Throwable throwable;
    
    private List<Localizable> msgPatterns;
    
    private List<Object[]> msgArguments;
    
    private Map<String, Object> context;

    
    public ExceptionContext(final Throwable throwable) {
        this.throwable = throwable;
        msgPatterns    = new ArrayList<Localizable>();
        msgArguments   = new ArrayList<Object[]>();
        context        = new HashMap<String, Object>();
    }

    
    public Throwable getThrowable() {
        return throwable;
    }

    
    public void addMessage(Localizable pattern,
                           Object ... arguments) {
        msgPatterns.add(pattern);
        msgArguments.add(ArgUtils.flatten(arguments));
    }

    
    public void setValue(String key, Object value) {
        context.put(key, value);
    }

    
    public Object getValue(String key) {
        return context.get(key);
    }

    
    public Set<String> getKeys() {
        return context.keySet();
    }

    
    public String getMessage() {
        return getMessage(Locale.US);
    }

    
    public String getLocalizedMessage() {
        return getMessage(Locale.getDefault());
    }

    
    public String getMessage(final Locale locale) {
        return buildMessage(locale, ": ");
    }

    
    public String getMessage(final Locale locale,
                             final String separator) {
        return buildMessage(locale, separator);
    }

    
    private String buildMessage(Locale locale,
                                String separator) {
        final StringBuilder sb = new StringBuilder();
        int count = 0;
        final int len = msgPatterns.size();
        for (int i = 0; i < len; i++) {
            final Localizable pat = msgPatterns.get(i);
            final Object[] args = msgArguments.get(i);
            final MessageFormat fmt = new MessageFormat(pat.getLocalizedString(locale),
                                                        locale);
            sb.append(fmt.format(args));
            if (++count < len) {
                // Add a separator if there are other messages.
                sb.append(separator);
            }
        }

        return sb.toString();
    }

    
    private void writeObject(ObjectOutputStream out)
        throws IOException {
        out.writeObject(throwable);
        serializeMessages(out);
        serializeContext(out);
    }
    
    private void readObject(ObjectInputStream in)
        throws IOException,
               ClassNotFoundException {
        throwable = (Throwable) in.readObject();
        deSerializeMessages(in);
        deSerializeContext(in);
    }

    
    private void serializeMessages(ObjectOutputStream out)
        throws IOException {
        // Step 1.
        final int len = msgPatterns.size();
        out.writeInt(len);
        // Step 2.
        for (int i = 0; i < len; i++) {
            final Localizable pat = msgPatterns.get(i);
            // Step 3.
            out.writeObject(pat);
            final Object[] args = msgArguments.get(i);
            final int aLen = args.length;
            // Step 4.
            out.writeInt(aLen);
            for (int j = 0; j < aLen; j++) {
                if (args[j] instanceof Serializable) {
                    // Step 5a.
                    out.writeObject(args[j]);
                } else {
                    // Step 5b.
                    out.writeObject(nonSerializableReplacement(args[j]));
                }
            }
        }
    }

    
    private void deSerializeMessages(ObjectInputStream in)
        throws IOException,
               ClassNotFoundException {
        // Step 1.
        final int len = in.readInt();
        msgPatterns = new ArrayList<Localizable>(len);
        msgArguments = new ArrayList<Object[]>(len);
        // Step 2.
        for (int i = 0; i < len; i++) {
            // Step 3.
            final Localizable pat = (Localizable) in.readObject();
            msgPatterns.add(pat);
            // Step 4.
            final int aLen = in.readInt();
            final Object[] args = new Object[aLen];
            for (int j = 0; j < aLen; j++) {
                // Step 5.
                args[j] = in.readObject();
            }
            msgArguments.add(args);
        }
    }

    
    private void serializeContext(ObjectOutputStream out)
        throws IOException {
        // Step 1.
        final int len = context.size();
        out.writeInt(len);
        for (Map.Entry<String, Object> entry : context.entrySet()) {
            // Step 2.
            out.writeObject(entry.getKey());
            final Object value = entry.getValue();
            if (value instanceof Serializable) {
                // Step 3a.
                out.writeObject(value);
            } else {
                // Step 3b.
                out.writeObject(nonSerializableReplacement(value));
            }
        }
    }

    
    private void deSerializeContext(ObjectInputStream in)
        throws IOException,
               ClassNotFoundException {
        // Step 1.
        final int len = in.readInt();
        context = new HashMap<String, Object>();
        for (int i = 0; i < len; i++) {
            // Step 2.
            final String key = (String) in.readObject();
            // Step 3.
            final Object value = in.readObject();
            context.put(key, value);
        }
    }

    
    private String nonSerializableReplacement(Object obj) {
        return "[Object could not be serialized: " + obj.getClass().getName() + "]";
    }
}
