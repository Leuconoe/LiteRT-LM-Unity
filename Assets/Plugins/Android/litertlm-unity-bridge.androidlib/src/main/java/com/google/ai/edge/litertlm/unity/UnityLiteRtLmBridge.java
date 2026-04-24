package com.google.ai.edge.litertlm.unity;

public final class UnityLiteRtLmBridge {
    static {
        System.loadLibrary("litertlm_jni");
    }

    private long handle;

    public synchronized boolean isInitialized() {
        return handle != 0L;
    }

    public synchronized void initialize(
            String modelPath,
            String backend,
            String cacheDir,
            int maxNumTokens,
            int maxNumImages,
            int cpuThreads,
            String systemInstruction) {
        if (handle != 0L) {
            throw new IllegalStateException("UnityLiteRtLmBridge is already initialized.");
        }

        handle = nativeCreateBridge(
                modelPath,
                backend,
                cacheDir,
                maxNumTokens,
                maxNumImages,
                cpuThreads,
                systemInstruction);
    }

    public synchronized String sendMessage(String text, String extraContextJson) {
        if (handle == 0L) {
            throw new IllegalStateException("UnityLiteRtLmBridge is not initialized.");
        }
        return nativeSendMessage(handle, text == null ? "" : text);
    }

    public synchronized void resetConversation(String systemInstruction) {
        if (handle == 0L) {
            throw new IllegalStateException("UnityLiteRtLmBridge is not initialized.");
        }
        nativeResetConversation(handle, systemInstruction == null ? "" : systemInstruction);
    }

    public synchronized void close() {
        if (handle != 0L) {
            nativeClose(handle);
            handle = 0L;
        }
    }

    public void setNativeMinLogSeverity(String level) {
        int severity;
        switch (level == null ? "ERROR" : level.toUpperCase()) {
            case "VERBOSE": severity = 0; break;
            case "DEBUG": severity = 1; break;
            case "INFO": severity = 2; break;
            case "WARNING": severity = 3; break;
            case "ERROR": severity = 4; break;
            case "FATAL": severity = 5; break;
            case "INFINITY": severity = 1000; break;
            default: throw new IllegalArgumentException("Unsupported log severity: " + level);
        }
        nativeSetMinLogSeverity(severity);
    }

    private static native long nativeCreateBridge(
            String modelPath,
            String backend,
            String cacheDir,
            int maxNumTokens,
            int maxNumImages,
            int cpuThreads,
            String systemInstruction);

    private static native String nativeSendMessage(long handle, String text);

    private static native void nativeResetConversation(long handle, String systemInstruction);

    private static native void nativeClose(long handle);

    private static native void nativeSetMinLogSeverity(int logSeverity);
}
