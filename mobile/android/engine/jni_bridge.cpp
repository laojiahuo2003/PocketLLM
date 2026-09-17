// JNI 桥接层：把 pocket C++ 引擎暴露给 Android (Kotlin/Java)
// 采用同步阻塞式 API，简单可靠：
//   nativeInit(path)      -> 加载 Model + Tokenizer（.pllm）
//   nativeGenerate(prompt) -> 生成一段回复文本
//   nativeRelease()        -> 释放全局实例
#include <jni.h>
#include <string>
#include <android/log.h>
#include "pocket.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "PocketLLM", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "PocketLLM", __VA_ARGS__)

namespace {
pocket::Model* g_model = nullptr;
pocket::Tokenizer* g_tok = nullptr;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_pocketllm_app_PocketEngine_nativeInit(JNIEnv* env, jobject, jstring path) {
    const char* cpath = env->GetStringUTFChars(path, nullptr);
    if (!cpath) return JNI_FALSE;

    // 释放上一次的实例（可重复 init 换模型）
    delete g_model; g_model = nullptr;
    delete g_tok;   g_tok = nullptr;

    auto* m = new (std::nothrow) pocket::Model();
    auto* t = new (std::nothrow) pocket::Tokenizer();
    bool ok = m && t && m->load(cpath) && t->load(cpath);
    env->ReleaseStringUTFChars(path, cpath);

    if (ok) {
        g_model = m;
        g_tok = t;
        LOGI("nativeInit ok: params=%llu", (unsigned long long)g_model->config().hidden_size
                ? g_model->vocab_size() : 0ULL);
        return JNI_TRUE;
    }
    delete m; delete t;
    LOGE("nativeInit failed");
    return JNI_FALSE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_pocketllm_app_PocketEngine_nativeGenerate(JNIEnv* env, jobject, jstring prompt, jint maxTokens) {
    if (!g_model || !g_tok) return env->NewStringUTF("");

    const char* cprompt = env->GetStringUTFChars(prompt, nullptr);
    std::string input = cprompt ? cprompt : "";
    env->ReleaseStringUTFChars(prompt, cprompt);

    pocket::GenerateConfig cfg;
    cfg.max_new_tokens = maxTokens;
    cfg.temperature = 0.8f;
    cfg.top_p = 0.9f;
    cfg.top_k = 50;

    std::string out;
    try {
        pocket::Generator gen(g_model, g_tok);
        out = gen.generate(input, cfg);
    } catch (const std::exception& e) {
        LOGE("generate exception: %s", e.what());
    } catch (...) {
        LOGE("generate unknown exception");
    }
    return env->NewStringUTF(out.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_pocketllm_app_PocketEngine_nativeRelease(JNIEnv*, jobject) {
    delete g_model; g_model = nullptr;
    delete g_tok;   g_tok = nullptr;
}