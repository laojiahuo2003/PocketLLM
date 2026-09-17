package com.pocketllm.app;

import android.content.Context;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * PocketLLM 引擎封装：负责把 assets 里的 .pllm 模型拷贝到私有目录，
 * 并用 JNI 加载 arm64 的 libpocket.so 引擎进行离线推理。
 *
 * 所有 native 调用都在独立单线程池上执行，避免阻塞 UI 线程。
 */
public class PocketEngine {

    static {
        System.loadLibrary("pocket");
    }

    /** 对应 C++ jni_bridge.cpp 里的 Java_com_pocketllm_app_PocketEngine_native 方法族 */
    private static native boolean nativeInit(String modelPath);
    private static native String nativeGenerate(String prompt, int maxTokens);
    private static native void nativeRelease();

    /** 模型资源名（已打进 APK 的 assets/ 下） */
    private static final String MODEL_ASSET = "pocket-0.1-sft-q8.pllm";

    public interface Callback {
        void onResult(String text);
    }

    private static volatile PocketEngine sInstance;

    private final Context context;
    private final ExecutorService pool = Executors.newSingleThreadExecutor();

    private PocketEngine(Context context) {
        this.context = context.getApplicationContext();
    }

    public static PocketEngine get(Context context) {
        if (sInstance == null) {
            synchronized (PocketEngine.class) {
                if (sInstance == null) {
                    sInstance = new PocketEngine(context);
                }
            }
        }
        return sInstance;
    }

    /** 把 assets 里的模型拷到 files/，返回绝对路径 */
    private String ensureModelFile() throws Exception {
        File dest = new File(context.getFilesDir(), MODEL_ASSET);
        if (!dest.exists()) {
            try (InputStream in = context.getAssets().open(MODEL_ASSET);
                 FileOutputStream out = new FileOutputStream(dest)) {
                byte[] buf = new byte[65536];
                int n;
                while ((n = in.read(buf)) > 0) {
                    out.write(buf, 0, n);
                }
            }
        }
        return dest.getAbsolutePath();
    }

    /** 异步加载模型，完成后回到主线程回调 */
    public void load(final Runnable onReady, final Runnable onError) {
        pool.execute(new Runnable() {
            @Override
            public void run() {
                boolean ok = false;
                try {
                    String path = ensureModelFile();
                    ok = nativeInit(path);
                } catch (Throwable t) {
                    t.printStackTrace();
                }
                runOnUi(ok ? onReady : onError);
            }
        });
    }

    /** 异步生成文本（同步内层推理，跑在子线程） */
    public void generate(final String prompt, final int maxTokens, final Callback cb) {
        pool.execute(new Runnable() {
            @Override
            public void run() {
                String out;
                try {
                    out = nativeGenerate(prompt == null ? "" : prompt, maxTokens);
                } catch (Throwable t) {
                    t.printStackTrace();
                    out = "[推理出错] " + t.getMessage();
                }
                final String result = out;
                new android.os.Handler(android.os.Looper.getMainLooper()).post(new Runnable() {
                    @Override
                    public void run() {
                        if (cb != null) cb.onResult(result);
                    }
                });
            }
        });
    }

    private void runOnUi(final Runnable r) {
        if (r == null) return;
        new android.os.Handler(android.os.Looper.getMainLooper()).post(r);
    }
}