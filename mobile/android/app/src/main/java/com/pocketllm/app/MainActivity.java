package com.pocketllm.app;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

/**
 * 极简离线聊天界面：输入问题 -> C++ 引擎本地推理 -> 显示回复。
 * 走纯 Android View，不依赖任何第三方 UI 库。
 */
public class MainActivity extends Activity {

    private PocketEngine engine;
    private TextView logView;
    private EditText input;
    private Button send;

    private final StringBuilder chatLog = new StringBuilder();

    private static final String PENDING_MARK = "🤖 思考中...\n\n";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        engine = PocketEngine.get(this);
        buildUI();
        loadModel();
    }

    private void buildUI() {
        // 外层垂直布局：上面是可滚动的聊天区(占满剩余空间)，下面固定输入行
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);

        // 聊天区：ScrollView 占满权重，内部放一个 wrap 高度的 TextView
        ScrollView scroll = new ScrollView(this);
        logView = new TextView(this);
        logView.setTextSize(16);
        logView.setPadding(24, 24, 24, 24);
        scroll.addView(logView, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT));
        root.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        // 输入行
        input = new EditText(this);
        input.setHint("输入你的杠精问题...");
        input.setSingleLine(true);

        send = new Button(this);
        send.setText("发送");
        send.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onSend();
            }
        });

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.addView(input, new LinearLayout.LayoutParams(0, -2, 1f));
        row.addView(send, new LinearLayout.LayoutParams(0, -2, 0.3f));
        root.addView(row);

        setContentView(root);
    }

    private void loadModel() {
        append("正在加载离线模型（Q8_0）...");
        engine.load(new Runnable() {
            @Override
            public void run() {
                Log.d("PocketUI", "load onReady");
                append("模型就绪 ✓ 开始聊天！");
            }
        }, new Runnable() {
            @Override
            public void run() {
                Log.e("PocketUI", "load onError");
                Toast.makeText(MainActivity.this, "模型加载失败", Toast.LENGTH_LONG).show();
                append("[加载失败] 请检查 APK 内是否含 .pllm");
            }
        });
    }

    // ---- 文件日志（避免 logcat 抓不到，供 run-as cat 排查）----
    private void fileLog(String msg) {
        try {
            java.io.FileOutputStream f = openFileOutput("log.txt", MODE_APPEND);
            f.write((System.currentTimeMillis() + " " + msg + "\n").getBytes());
            f.close();
        } catch (Exception e) { }
    }

    private void onSend() {
        final String q = input.getText().toString().trim();
        fileLog("onSend q=[" + q + "] enabled=" + send.isEnabled());
        if (q.isEmpty() || send.isEnabled() == false) {
            fileLog("onSend EARLY-RETURN");
            return;
        }
        input.setText("");
        append("👤 " + q);
        append("🤖 思考中...");
        render();
        fileLog("echo rendered, calling generate");
        send.setEnabled(false);
        engine.generate(q, 60, new PocketEngine.Callback() {
            @Override
            public void onResult(String text) {
                fileLog("onResult len=" + (text == null ? -1 : text.length()) + " text=[" + text + "]");
                send.setEnabled(true);
                pendingToReply("🤖 " + (text == null ? "[空]" : text.trim()));
                fileLog("reply rendered");
            }
        });
    }

    // ---- 渲染辅助 ----
    private void render() {
        logView.setText(chatLog);
    }

    private void append(String s) {
        chatLog.append(s).append("\n\n");
        render();
    }

    /** 把最后一条 "思考中" 换成真实回复 */
    private void pendingToReply(String reply) {
        int idx = chatLog.lastIndexOf(PENDING_MARK);
        if (idx >= 0) {
            chatLog.delete(idx, chatLog.length());
        }
        chatLog.append(reply).append("\n\n");
        render();
    }
}