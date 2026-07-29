using System;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Implemented by any sample component that keeps a loaded model alive.
    ///
    /// A LiteRT-LM engine holds native memory that the C# garbage collector does
    /// not account for — a 2.6 GB multimodal model stays resident until the
    /// engine is disposed. Scene switching must therefore release explicitly
    /// rather than wait for finalization, otherwise two models are resident at
    /// once and the device can hit the low-memory killer.
    /// </summary>
    public interface ILiteRtLmModelHost
    {
        /// <summary>
        /// Releases every engine this component owns and stops any background
        /// work that uses them. Must be safe to call more than once, and safe to
        /// call when nothing was ever loaded.
        /// </summary>
        void ReleaseModels();
    }

    /// <summary>Helpers for releasing every model loaded in the current scene.</summary>
    public static class LiteRtLmModelMemory
    {
        /// <summary>
        /// Calls <see cref="ILiteRtLmModelHost.ReleaseModels"/> on every host in the
        /// loaded scenes, then asks Unity to drop unused assets. Call this before
        /// loading another scene so the outgoing model is gone before the next one
        /// is created.
        /// </summary>
        /// <returns>Number of hosts released.</returns>
        public static int ReleaseAll()
        {
            var released = 0;

#if UNITY_2022_2_OR_NEWER
            var behaviours = UnityEngine.Object.FindObjectsByType<MonoBehaviour>(
                FindObjectsInactive.Include, FindObjectsSortMode.None);
#else
            var behaviours = UnityEngine.Object.FindObjectsOfType<MonoBehaviour>(true);
#endif
            foreach (var behaviour in behaviours)
            {
                if (behaviour is not ILiteRtLmModelHost host)
                {
                    continue;
                }

                try
                {
                    host.ReleaseModels();
                    released++;
                }
                catch (Exception ex)
                {
                    // One misbehaving host must not block the others from releasing.
                    Debug.LogError($"[LiteRT-LM] ReleaseModels failed on {behaviour.GetType().Name}: {ex.Message}");
                }
            }

            Resources.UnloadUnusedAssets();
            GC.Collect();
            GC.WaitForPendingFinalizers();

            return released;
        }
    }
}
