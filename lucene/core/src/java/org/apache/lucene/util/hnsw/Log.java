package org.apache.lucene.util.hnsw;

import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class Log {
  private OutputStreamWriter out;

  public Log() {
    try {
      out = new FileWriter("/home/jonathan/Projects/lucene/tests.log", true);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void info(String s) {
    try {
      out.write(s + "\n");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void flush() {
    try {
      out.flush();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
