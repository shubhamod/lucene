package org.apache.lucene.util.hnsw;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.RandomAccessFile;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class Log {
  private final RandomAccessFile out;
  private final File logFile;
  private long lastEntryPosition = 0;

  public Log() {
    try {
      String tempDir = System.getProperty("user.home") + "/logs";
      String timeStamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"));
      logFile = new File(tempDir, "lucene_tests_" + timeStamp + ".log");
      System.out.println("Logging to " + logFile.getAbsolutePath());
      System.out.flush();
      out = new RandomAccessFile(logFile, "rw");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public void info(String s) {
    try {
      out.writeBytes(String.format("%s %s%n", Thread.currentThread().getId(), s));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void flush() {
    try {
      out.getFD().sync();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void writeLastEntryToStdout() {
    try {
      flush();
      RandomAccessFile raf = new RandomAccessFile(logFile, "r");
      raf.seek(lastEntryPosition);
      String line;
      System.out.println("Last entry from " + logFile + ":");
      while ((line = raf.readLine()) != null) {
        System.out.println(line);
      }
      raf.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public void begin(String formatted) {
    try {
      lastEntryPosition = out.getFilePointer();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    info(formatted);
//    System.out.println(formatted);
  }
}
