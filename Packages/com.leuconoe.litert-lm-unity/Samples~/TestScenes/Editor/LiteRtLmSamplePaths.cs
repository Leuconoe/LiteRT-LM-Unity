using System.IO;
using UnityEditor;
using UnityEngine;

namespace LiteRTLM.Unity.Editor
{
    /// <summary>
    /// Resolves the sample's own asset folders at editor time.
    ///
    /// The sample is imported through Package Manager, so it lands under
    /// <c>Assets/Samples/&lt;package&gt;/&lt;version&gt;/Test Scenes/</c> — a path that
    /// depends on the package version and on the user's project. Nothing here may
    /// hard-code it. Every folder is derived from where this script actually sits.
    /// </summary>
    public static class LiteRtLmSamplePaths
    {
        /// <summary>Used only when this script cannot be located (should not happen in a normal import).</summary>
        private const string LegacyScenesFolder = "Assets/Scenes/Tests";

        private static string s_cachedRoot;

        /// <summary>Sample root, i.e. the folder that contains <c>Editor/</c>, <c>Runtime/</c> and <c>Scenes/</c>.</summary>
        public static string SampleRoot
        {
            get
            {
                if (!string.IsNullOrEmpty(s_cachedRoot))
                {
                    return s_cachedRoot;
                }

                var guids = AssetDatabase.FindAssets($"{nameof(LiteRtLmSamplePaths)} t:MonoScript");
                foreach (var guid in guids)
                {
                    var scriptPath = AssetDatabase.GUIDToAssetPath(guid);
                    if (string.IsNullOrEmpty(scriptPath) ||
                        Path.GetFileNameWithoutExtension(scriptPath) != nameof(LiteRtLmSamplePaths))
                    {
                        continue;
                    }

                    // <root>/Editor/LiteRtLmSamplePaths.cs -> <root>
                    var editorFolder = Path.GetDirectoryName(scriptPath);
                    var root = Path.GetDirectoryName(editorFolder);
                    if (!string.IsNullOrEmpty(root))
                    {
                        s_cachedRoot = root.Replace('\\', '/');
                        return s_cachedRoot;
                    }
                }

                Debug.LogWarning(
                    $"[LiteRT-LM] Could not locate {nameof(LiteRtLmSamplePaths)}; falling back to {LegacyScenesFolder}.");
                s_cachedRoot = null;
                return null;
            }
        }

        /// <summary>Folder holding the sample scenes. Created on demand.</summary>
        public static string ScenesFolder
        {
            get
            {
                var root = SampleRoot;
                return string.IsNullOrEmpty(root) ? LegacyScenesFolder : $"{root}/Scenes";
            }
        }

        /// <summary>Folder for throwaway build scenes generated during an APK build.</summary>
        public static string GeneratedFolder => $"{ScenesFolder}/Generated";

        /// <summary>Asset path of a sample scene by file name (without extension).</summary>
        public static string Scene(string sceneName) => $"{ScenesFolder}/{sceneName}.unity";

        /// <summary>Asset path of a generated build scene by file name (without extension).</summary>
        public static string GeneratedScene(string sceneName) => $"{GeneratedFolder}/{sceneName}.unity";

        /// <summary>Creates a folder (and its parents) under Assets if it does not exist yet.</summary>
        public static void EnsureFolder(string assetFolderPath)
        {
            if (string.IsNullOrEmpty(assetFolderPath) || AssetDatabase.IsValidFolder(assetFolderPath))
            {
                return;
            }

            var parent = Path.GetDirectoryName(assetFolderPath)?.Replace('\\', '/');
            var leaf = Path.GetFileName(assetFolderPath);
            if (string.IsNullOrEmpty(parent) || string.IsNullOrEmpty(leaf))
            {
                return;
            }

            EnsureFolder(parent);
            AssetDatabase.CreateFolder(parent, leaf);
        }
    }
}
