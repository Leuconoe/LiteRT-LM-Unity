using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
#if UNITY_ANDROID && !UNITY_EDITOR
using UnityEngine.Android;
#endif

namespace LiteRTLM.Unity
{
    // Live microphone capture with streaming energy-VAD endpointing.
    //
    // The VAD mirrors the native dual-mode VAD v2 "energy" parameters
    // (litertlm-unity-bridge take7) so a mic-captured utterance behaves like
    // a pre-trimmed file clip:
    //   - adaptive noise floor calibrated from the first ~300 ms after start
    //   - speech-on at noiseFloor + 9 dB, speech-off at noiseFloor + 6 dB
    //     (hysteresis)
    //   - 210 ms hangover of continuous silence before endpointing
    //   - 90 ms preroll retained before the detected speech onset
    //   - utterances with < 200 ms of voiced audio are discarded as false
    //     triggers
    //   - 8 s max utterance safety cutoff
    //
    // Because the emitted utterance is already endpointed here, the native
    // ASR preprocessing vadMode can stay "energy": the native trim only
    // re-trims the edges of an utterance this component already bounded, and
    // both trims are conservative (preroll/hangover padded), so they compose
    // safely.
    //
    // Android manifest note: there is no AndroidManifest.xml under
    // Assets/Plugins/Android — Unity injects android.permission.RECORD_AUDIO
    // automatically because this script references UnityEngine.Microphone.
    // The runtime permission prompt is handled in StartListening() via
    // UnityEngine.Android.Permission.
    public sealed class LiteRtLmMicVadCapture : MonoBehaviour
    {
        public enum MicVadState
        {
            Idle,
            RequestingPermission,
            Calibrating,
            Listening,
            Speech,
        }

        public const int TargetSampleRate = 16000;

        private const int LoopBufferSeconds = 10;
        private const int FrameMs = 30;
        private const int NoiseCalibrationMs = 300;
        private const float SpeechOnDeltaDb = 9f;
        private const float SpeechOffDeltaDb = 6f;
        private const int HangoverMs = 210;
        private const int PrerollMs = 90;
        private const int MinSpeechMs = 200;
        private const int MaxUtteranceMs = 8000;
        private const float MinNoiseFloorDb = -75f;
        private const float SilenceFloorDb = -90f;

        // Fired on the main thread with the endpointed utterance as mono
        // float PCM resampled to 16 kHz (preroll + speech + hangover tail).
        public event Action<float[]> OnUtteranceCaptured;
        public event Action<string> OnCaptureError;
        public event Action<MicVadState> OnStateChanged;

        public MicVadState State { get; private set; } = MicVadState.Idle;
        public float CurrentLevelDb { get; private set; } = SilenceFloorDb;
        public float NoiseFloorDb { get; private set; } = MinNoiseFloorDb;
        public float SpeechOnThresholdDb => NoiseFloorDb + SpeechOnDeltaDb;
        public bool IsCapturing => State != MicVadState.Idle;
        public float UtteranceSeconds => _utteranceMs / 1000f;
        public string DeviceName { get; private set; } = string.Empty;

        private AudioClip _clip;
        private int _captureRate = TargetSampleRate;
        private int _frameSamples;
        private int _readPosition;
        private float[] _frameScratch;

        private readonly Queue<float[]> _prerollFrames = new Queue<float[]>();
        private int _prerollFrameCapacity;
        private readonly List<float> _utterance = new List<float>();

        private float _calibrationDbSum;
        private int _calibrationFrameCount;
        private int _calibrationFramesTarget;
        private int _silenceRunMs;
        private int _speechMs;
        private int _utteranceMs;

        public void StartListening()
        {
            if (IsCapturing)
            {
                return;
            }

#if UNITY_ANDROID && !UNITY_EDITOR
            if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
            {
                SetState(MicVadState.RequestingPermission);
                var callbacks = new PermissionCallbacks();
                callbacks.PermissionGranted += _ => BeginCapture();
                callbacks.PermissionDenied += _ =>
                {
                    SetState(MicVadState.Idle);
                    RaiseError("RECORD_AUDIO permission denied.");
                };
                Permission.RequestUserPermission(Permission.Microphone, callbacks);
                return;
            }
#endif
            BeginCapture();
        }

        public void StopListening()
        {
            if (_clip != null || Microphone.IsRecording(DeviceName))
            {
                Microphone.End(DeviceName);
            }

            if (_clip != null)
            {
                Destroy(_clip);
                _clip = null;
            }

            SetState(MicVadState.Idle);
            CurrentLevelDb = SilenceFloorDb;
        }

        private void BeginCapture()
        {
            var devices = Microphone.devices;
            if (devices == null || devices.Length == 0)
            {
                SetState(MicVadState.Idle);
                RaiseError("No microphone device available.");
                return;
            }

            // Empty string selects the platform default microphone.
            DeviceName = string.Empty;
            Microphone.GetDeviceCaps(DeviceName, out var minFreq, out var maxFreq);
            _captureRate = TargetSampleRate;
            if (maxFreq > 0 && _captureRate > maxFreq)
            {
                _captureRate = maxFreq;
            }

            if (minFreq > 0 && _captureRate < minFreq)
            {
                _captureRate = minFreq;
            }

            _clip = Microphone.Start(DeviceName, true, LoopBufferSeconds, _captureRate);
            if (_clip == null)
            {
                SetState(MicVadState.Idle);
                RaiseError("Microphone.Start failed (no clip returned).");
                return;
            }

            _frameSamples = Mathf.Max(1, _captureRate * FrameMs / 1000);
            _frameScratch = new float[_frameSamples];
            _prerollFrameCapacity = Mathf.Max(1, PrerollMs / FrameMs);
            _readPosition = 0;
            ResetVadState();
            SetState(MicVadState.Calibrating);
        }

        private void ResetVadState()
        {
            _prerollFrames.Clear();
            _utterance.Clear();
            _calibrationDbSum = 0f;
            _calibrationFrameCount = 0;
            _calibrationFramesTarget = Mathf.Max(1, NoiseCalibrationMs / FrameMs);
            _silenceRunMs = 0;
            _speechMs = 0;
            _utteranceMs = 0;
            NoiseFloorDb = MinNoiseFloorDb;
            CurrentLevelDb = SilenceFloorDb;
        }

        private void Update()
        {
            if (_clip == null ||
                State == MicVadState.Idle ||
                State == MicVadState.RequestingPermission)
            {
                return;
            }

            var clipSamples = _clip.samples;
            var position = Microphone.GetPosition(DeviceName);
            if (position < 0 || clipSamples <= 0)
            {
                return;
            }

            var available = position - _readPosition;
            if (available < 0)
            {
                available += clipSamples;
            }

            while (available >= _frameSamples && _clip != null)
            {
                ReadFrame(clipSamples);
                available -= _frameSamples;
                ProcessFrame(_frameScratch);
            }
        }

        private void ReadFrame(int clipSamples)
        {
            if (_readPosition + _frameSamples <= clipSamples)
            {
                _clip.GetData(_frameScratch, _readPosition);
            }
            else
            {
                // The frame wraps around the loop buffer boundary
                // (roughly once every LoopBufferSeconds).
                var headCount = clipSamples - _readPosition;
                var tailCount = _frameSamples - headCount;
                var head = new float[headCount];
                var tail = new float[tailCount];
                _clip.GetData(head, _readPosition);
                _clip.GetData(tail, 0);
                Array.Copy(head, 0, _frameScratch, 0, headCount);
                Array.Copy(tail, 0, _frameScratch, headCount, tailCount);
            }

            _readPosition = (_readPosition + _frameSamples) % clipSamples;
        }

        private void ProcessFrame(float[] frame)
        {
            var frameDb = ComputeFrameDb(frame);
            CurrentLevelDb = frameDb;

            switch (State)
            {
                case MicVadState.Calibrating:
                    _calibrationDbSum += frameDb;
                    _calibrationFrameCount++;
                    PushPrerollFrame(frame);
                    if (_calibrationFrameCount >= _calibrationFramesTarget)
                    {
                        NoiseFloorDb = Mathf.Max(
                            _calibrationDbSum / _calibrationFrameCount,
                            MinNoiseFloorDb);
                        SetState(MicVadState.Listening);
                    }

                    break;

                case MicVadState.Listening:
                    PushPrerollFrame(frame);
                    if (frameDb > NoiseFloorDb + SpeechOnDeltaDb)
                    {
                        EnterSpeech(frame);
                    }
                    else
                    {
                        // Slow adaptive noise floor: track downward quickly,
                        // upward slowly so speech onsets never raise the gate.
                        var adaptRate = frameDb < NoiseFloorDb ? 0.2f : 0.02f;
                        NoiseFloorDb = Mathf.Max(
                            Mathf.Lerp(NoiseFloorDb, frameDb, adaptRate),
                            MinNoiseFloorDb);
                    }

                    break;

                case MicVadState.Speech:
                    AppendUtteranceFrame(frame);
                    _utteranceMs += FrameMs;
                    if (frameDb < NoiseFloorDb + SpeechOffDeltaDb)
                    {
                        _silenceRunMs += FrameMs;
                    }
                    else
                    {
                        _silenceRunMs = 0;
                        _speechMs += FrameMs;
                    }

                    if (_silenceRunMs >= HangoverMs || _utteranceMs >= MaxUtteranceMs)
                    {
                        Endpoint();
                    }

                    break;
            }
        }

        private void EnterSpeech(float[] onsetFrame)
        {
            _utterance.Clear();
            _utteranceMs = 0;
            // The preroll ring includes the onset frame itself (pushed just
            // before the gate check), so drop the newest entry and append the
            // onset frame explicitly to avoid duplicating it.
            var prerollToKeep = _prerollFrames.Count - 1;
            foreach (var prerollFrame in _prerollFrames)
            {
                if (prerollToKeep-- <= 0)
                {
                    break;
                }

                _utterance.AddRange(prerollFrame);
                _utteranceMs += FrameMs;
            }

            _prerollFrames.Clear();
            AppendUtteranceFrame(onsetFrame);
            _utteranceMs += FrameMs;
            _speechMs = FrameMs;
            _silenceRunMs = 0;
            SetState(MicVadState.Speech);
        }

        private void Endpoint()
        {
            if (_speechMs < MinSpeechMs)
            {
                // False trigger (door click, cough tail, ...): discard and
                // keep listening.
                _utterance.Clear();
                _utteranceMs = 0;
                _speechMs = 0;
                _silenceRunMs = 0;
                SetState(MicVadState.Listening);
                return;
            }

            var pcm = _utterance.ToArray();
            if (_captureRate != TargetSampleRate)
            {
                pcm = ResampleLinear(pcm, _captureRate, TargetSampleRate);
            }

            // Single-shot capture: stop the microphone before handing the
            // utterance out so a heavyweight ASR call cannot overflow the
            // loop buffer while it runs.
            StopListening();
            OnUtteranceCaptured?.Invoke(pcm);
        }

        private void PushPrerollFrame(float[] frame)
        {
            var copy = new float[frame.Length];
            Array.Copy(frame, copy, frame.Length);
            _prerollFrames.Enqueue(copy);
            while (_prerollFrames.Count > _prerollFrameCapacity)
            {
                _prerollFrames.Dequeue();
            }
        }

        private void AppendUtteranceFrame(float[] frame)
        {
            for (var i = 0; i < frame.Length; i++)
            {
                _utterance.Add(frame[i]);
            }
        }

        private static float ComputeFrameDb(float[] frame)
        {
            double sumSquares = 0.0;
            for (var i = 0; i < frame.Length; i++)
            {
                sumSquares += (double)frame[i] * frame[i];
            }

            var rms = Math.Sqrt(sumSquares / Math.Max(1, frame.Length));
            var db = 20f * (float)Math.Log10(Math.Max(rms, 1e-6));
            return Mathf.Max(db, SilenceFloorDb);
        }

        private static float[] ResampleLinear(float[] input, int fromRate, int toRate)
        {
            if (input.Length == 0 || fromRate == toRate)
            {
                return input;
            }

            var outputLength = (int)((long)input.Length * toRate / fromRate);
            var output = new float[Math.Max(outputLength, 1)];
            var step = (double)fromRate / toRate;
            for (var i = 0; i < output.Length; i++)
            {
                var sourceIndex = i * step;
                var index0 = (int)sourceIndex;
                if (index0 >= input.Length - 1)
                {
                    output[i] = input[input.Length - 1];
                    continue;
                }

                var t = (float)(sourceIndex - index0);
                output[i] = input[index0] + (input[index0 + 1] - input[index0]) * t;
            }

            return output;
        }

        // Minimal PCM16 mono WAV writer for handing mic captures to the
        // file-path-based native ASR smoke entry points.
        public static void WriteWav16BitMono(string path, float[] samples, int sampleRate)
        {
            if (samples == null)
            {
                throw new ArgumentNullException(nameof(samples));
            }

            using var stream = new FileStream(path, FileMode.Create, FileAccess.Write);
            using var writer = new BinaryWriter(stream);
            var dataBytes = samples.Length * 2;
            writer.Write(Encoding.ASCII.GetBytes("RIFF"));
            writer.Write(36 + dataBytes);
            writer.Write(Encoding.ASCII.GetBytes("WAVE"));
            writer.Write(Encoding.ASCII.GetBytes("fmt "));
            writer.Write(16);
            writer.Write((short)1);           // PCM
            writer.Write((short)1);           // mono
            writer.Write(sampleRate);
            writer.Write(sampleRate * 2);     // byte rate
            writer.Write((short)2);           // block align
            writer.Write((short)16);          // bits per sample
            writer.Write(Encoding.ASCII.GetBytes("data"));
            writer.Write(dataBytes);
            for (var i = 0; i < samples.Length; i++)
            {
                var clamped = Mathf.Clamp(samples[i], -1f, 1f);
                writer.Write((short)Mathf.RoundToInt(clamped * 32767f));
            }
        }

        // Logged transitions keep the state machine observable through
        // logcat on devices where the GL surface is not screencap-visible.
        private void SetState(MicVadState next)
        {
            if (State == next)
            {
                return;
            }

            var previous = State;
            State = next;
            Debug.Log($"LiteRtLmMicVadCapture state: {previous} -> {next} (noiseFloor={NoiseFloorDb:0.0} dB, level={CurrentLevelDb:0.0} dB)");
            OnStateChanged?.Invoke(next);
        }

        private void RaiseError(string message)
        {
            Debug.LogWarning($"LiteRtLmMicVadCapture: {message}");
            OnCaptureError?.Invoke(message);
        }

        private void OnDisable()
        {
            StopListening();
        }
    }
}
