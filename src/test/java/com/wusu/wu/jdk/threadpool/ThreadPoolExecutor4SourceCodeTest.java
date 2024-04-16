package com.wusu.wu.jdk.threadpool;

import org.junit.Before;
import org.junit.Test;

import java.util.Random;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.Assert.*;

public class ThreadPoolExecutor4SourceCodeTest {

    private ThreadPoolExecutor4SourceCode threadPoolExecutor4SourceCode;

    @Before
    public void setUp() throws Exception {
        threadPoolExecutor4SourceCode = new ThreadPoolExecutor4SourceCode(10, 20, 1, TimeUnit.MINUTES,
                new ArrayBlockingQueue<>(10),
                Executors.defaultThreadFactory(),
                new ThreadPoolExecutor4SourceCode.CallerRunsPolicy());
    }


    @Test
    public void testRejectedExecution() {
        for (int i = 0; i < 50; i++) {
            int finalI = i;
            threadPoolExecutor4SourceCode.execute(() -> {
                try {
                    System.out.println(Thread.currentThread().getName()+" is running: i = " + finalI);
                    Thread.sleep(new Random().nextInt(100));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
        }
    }
}