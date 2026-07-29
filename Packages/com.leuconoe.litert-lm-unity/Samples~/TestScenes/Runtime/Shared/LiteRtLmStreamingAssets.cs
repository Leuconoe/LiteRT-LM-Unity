using System;
using System.IO;
using UnityEngine;

namespace LiteRTLM.Unity
{
    /// <summary>
    /// Locates model and asset files under StreamingAssets on desktop.
    ///
    /// Scenes serialize their model path, so a scene authored before the
    /// StreamingAssets reorganisation still carries the old flat path
    /// (`gemma-4-E2B-it.litertlm`) even after the script default was fixed to
    /// `Multimodal/gemma-4-e2b/gemma-4-E2B-it.litertlm`. Rather than hand-edit
    /// every scene, resolution falls back to a recursive search by file name, so
    /// a stale or relocated entry still finds the model.
    ///
    /// Android is not handled here: there StreamingAssets lives inside the APK and
    /// must be read through UnityWebRequest, which the ASR runner already does.
    /// </summary>
    public static class LiteRtLmStreamingAssets
    {
        /// <summary>
        /// Absolute path for a StreamingAssets-relative entry, or null when it
        /// cannot be found. <paramref name="howResolved"/> reports whether the
        /// original path was used or a fallback search found it elsewhere.
        /// </summary>
        public static string Resolve(string relativePath, out string howResolved)
        {
            howResolved = null;

            if (string.IsNullOrWhiteSpace(relativePath))
            {
                return null;
            }

            var root = Application.streamingAssetsPath;
            var direct = Path.Combine(root, relativePath);
            if (File.Exists(direct))
            {
                howResolved = "exact";
                return direct;
            }

            var fileName = Path.GetFileName(relativePath);
            if (string.IsNullOrEmpty(fileName) || !Directory.Exists(root))
            {
                return null;
            }

            try
            {
                var matches = Directory.GetFiles(root, fileName, SearchOption.AllDirectories);
                if (matches.Length > 0)
                {
                    howResolved = matches.Length == 1
                        ? $"relocated to {Relative(matches[0])}"
                        : $"relocated to {Relative(matches[0])} ({matches.Length} candidates)";
                    return matches[0];
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[LiteRT-LM] StreamingAssets search failed for '{fileName}': {ex.Message}");
            }

            return null;
        }

        /// <summary>Convenience overload when the caller does not need the reason.</summary>
        public static string Resolve(string relativePath) => Resolve(relativePath, out _);

        /// <summary>
        /// Same as <see cref="Resolve(string,out string)"/> but throws with a
        /// message that lists what actually is available, which is far more useful
        /// than "file not found" when a model simply has not been downloaded.
        /// </summary>
        public static string ResolveOrThrow(string relativePath)
        {
            var resolved = Resolve(relativePath, out var how);
            if (resolved != null)
            {
                if (how != null && how != "exact")
                {
                    Debug.Log($"[LiteRT-LM] '{relativePath}' {how}.");
                }

                return resolved;
            }

            throw new FileNotFoundException(
                $"Model '{relativePath}' was not found under StreamingAssets. " +
                $"Available model folders: {DescribeAvailable()}. " +
                "Models are not shipped with the package — see the README download tables.",
                relativePath);
        }

        /// <summary>Short listing of what is present, for error messages.</summary>
        public static string DescribeAvailable()
        {
            var root = Application.streamingAssetsPath;
            if (!Directory.Exists(root))
            {
                return "(StreamingAssets folder missing)";
            }

            try
            {
                var dirs = Directory.GetDirectories(root);
                if (dirs.Length == 0)
                {
                    return "(none)";
                }

                var names = new string[dirs.Length];
                for (var i = 0; i < dirs.Length; i++)
                {
                    names[i] = Path.GetFileName(dirs[i]);
                }

                return string.Join(", ", names);
            }
            catch (Exception ex)
            {
                return $"(unreadable: {ex.Message})";
            }
        }

        private static string Relative(string absolutePath)
        {
            var root = Application.streamingAssetsPath.Replace('\\', '/');
            var path = absolutePath.Replace('\\', '/');
            return path.StartsWith(root, StringComparison.OrdinalIgnoreCase)
                ? path.Substring(root.Length).TrimStart('/')
                : path;
        }
    }
}
