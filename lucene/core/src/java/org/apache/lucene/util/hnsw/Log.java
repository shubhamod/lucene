package org.apache.lucene.util.hnsw;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Log {
  private final BufferedWriter out;

  public Log() {
    try {
      String tempDir = System.getProperty("java.io.tmpdir");
      String timeStamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
      File logFile = new File(tempDir, "lucene_tests_" + timeStamp + ".log");
      System.out.println("Logging to " + logFile.getAbsolutePath());
      out = new BufferedWriter(new FileWriter(logFile, true));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void info(String s) {
    try {
      System.out.println(s);
      out.write(String.format("%s %s%n", Thread.currentThread().getId(), s));
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
