<#
.SYNOPSIS
Check that the C# Supertonic text front end produces the same ids as the Python
reference implementation.

.DESCRIPTION
The front end is a port, and a port that drifts feeds the model different input
than the bench that validated it — silently, as slightly wrong pronunciation
rather than as an error. This compiles LiteRtLmSupertonicText.cs on its own and
compares its output, case by case, against ids dumped from the reference.

Ground truth comes from Tools/Windows/TtsBench/dump_reference_text_ids.py; this
script regenerates it when the file is missing.

.EXAMPLE
  .\Tools\Windows\Test-SupertonicTextParity.ps1
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$AssetsDir = "",
    [string]$ReferenceJson = "",
    [string]$IndexerJson = "",
    [switch]$Regenerate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BenchDir = Join-Path $PSScriptRoot "TtsBench"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

if (!$AssetsDir) { $AssetsDir = Join-Path $ProjectRoot "External\tts-work\supertonic-2-fp32" }
if (!$ReferenceJson) { $ReferenceJson = Join-Path $ProjectRoot "External\tts-work\reference-text-ids.json" }
if (!$IndexerJson) { $IndexerJson = Join-Path $AssetsDir "unicode_indexer.json" }

if ($Regenerate -or !(Test-Path $ReferenceJson)) {
    $venvPython = Join-Path $BenchDir ".venv-convert\Scripts\python.exe"
    if (!(Test-Path $venvPython)) { throw "Conversion venv missing; run Convert-SupertonicToLiteRt.ps1 -Bootstrap." }
    & $venvPython (Join-Path $BenchDir "dump_reference_text_ids.py") --assets-dir $AssetsDir --out $ReferenceJson
    if ($LASTEXITCODE -ne 0) { throw "reference dump failed." }
}
if (!(Test-Path $IndexerJson)) { throw "unicode_indexer.json not found: $IndexerJson" }

# Compile the front end alone against netstandard — it has no Unity dependency,
# which is what makes this testable outside the editor.
$csc = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\Roslyn\csc.exe"
) + (Get-ChildItem "C:\Program Files*\Microsoft Visual Studio\*\*\MSBuild\Current\Bin\Roslyn\csc.exe" -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }) |
    Where-Object { Test-Path $_ } | Select-Object -First 1
if (!$csc) { throw "No Roslyn csc.exe found." }

$source = Join-Path $ProjectRoot "Packages\com.leuconoe.litert-lm-unity\Runtime\LiteRtLmSupertonicText.cs"
$harness = Join-Path $env:TEMP "SupertonicTextParityHarness.cs"
$exe = Join-Path $env:TEMP "SupertonicTextParity.exe"

@'
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using LiteRTLM.Unity;

internal static class Harness
{
    private static int Main(string[] args)
    {
        Console.OutputEncoding = Encoding.UTF8;
        var indexerPath = args[0];
        var referencePath = args[1];

        var encoder = LiteRtLmSupertonicText.FromIndexerJson(File.ReadAllText(indexerPath));
        var reference = File.ReadAllText(referencePath, Encoding.UTF8);

        var failures = 0;
        var cases = 0;
        foreach (var block in SplitObjects(reference))
        {
            cases++;
            var text = Unescape(Field(block, "text"));
            var lang = Field(block, "lang");
            var expected = Numbers(block, "ids");

            int[] ids;
            float[] mask;
            try
            {
                encoder.Encode(text, lang, out ids, out mask);
            }
            catch (Exception exception)
            {
                Console.WriteLine("FAIL  " + text + "  -> " + exception.Message);
                failures++;
                continue;
            }

            if (ids.Length != expected.Count)
            {
                Console.WriteLine("FAIL  " + text + "  length " + ids.Length + " != " + expected.Count);
                failures++;
                continue;
            }

            var mismatch = -1;
            for (var i = 0; i < ids.Length; i++)
            {
                if (ids[i] != expected[i]) { mismatch = i; break; }
            }

            if (mismatch >= 0)
            {
                Console.WriteLine("FAIL  " + text + "  id[" + mismatch + "] " + ids[mismatch] + " != " + expected[mismatch]);
                failures++;
            }
            else
            {
                Console.WriteLine("PASS  " + lang + "  " + ids.Length + " ids  " + text);
            }
        }

        Console.WriteLine();
        Console.WriteLine(failures == 0
            ? cases + " case(s) match the reference."
            : failures + " of " + cases + " case(s) differ.");
        return failures == 0 ? 0 : 1;
    }

    private static IEnumerable<string> SplitObjects(string json)
    {
        var depth = 0; var start = -1; var inString = false; var escape = false;
        for (var i = 0; i < json.Length; i++)
        {
            var c = json[i];
            if (escape) { escape = false; continue; }
            if (c == '\\' && inString) { escape = true; continue; }
            if (c == '"') { inString = !inString; continue; }
            if (inString) continue;
            if (c == '{') { if (depth == 0) start = i; depth++; }
            else if (c == '}') { depth--; if (depth == 0 && start >= 0) yield return json.Substring(start, i - start + 1); }
        }
    }

    private static string Field(string block, string name)
    {
        var key = "\"" + name + "\"";
        var at = block.IndexOf(key, StringComparison.Ordinal);
        if (at < 0) return string.Empty;
        var colon = block.IndexOf(':', at + key.Length);
        var quote = block.IndexOf('"', colon + 1);
        var builder = new StringBuilder();
        for (var i = quote + 1; i < block.Length; i++)
        {
            if (block[i] == '\\') { builder.Append(block[i]); builder.Append(block[i + 1]); i++; continue; }
            if (block[i] == '"') break;
            builder.Append(block[i]);
        }
        return builder.ToString();
    }

    private static string Unescape(string value)
    {
        var builder = new StringBuilder(value.Length);
        for (var i = 0; i < value.Length; i++)
        {
            if (value[i] != '\\') { builder.Append(value[i]); continue; }
            i++;
            switch (value[i])
            {
                case 'n': builder.Append('\n'); break;
                case 't': builder.Append('\t'); break;
                case 'r': builder.Append('\r'); break;
                case 'u':
                    builder.Append((char)Convert.ToInt32(value.Substring(i + 1, 4), 16));
                    i += 4;
                    break;
                default: builder.Append(value[i]); break;
            }
        }
        return builder.ToString();
    }

    private static List<int> Numbers(string block, string name)
    {
        var result = new List<int>();
        var key = "\"" + name + "\"";
        var at = block.IndexOf(key, StringComparison.Ordinal);
        if (at < 0) return result;
        var open = block.IndexOf('[', at);
        var close = block.IndexOf(']', open);
        var value = 0; var negative = false; var inNumber = false;
        for (var i = open + 1; i < close; i++)
        {
            var c = block[i];
            if (c == '-' && !inNumber) { negative = true; inNumber = true; value = 0; }
            else if (c >= '0' && c <= '9') { inNumber = true; value = value * 10 + (c - '0'); }
            else if (inNumber) { result.Add(negative ? -value : value); value = 0; negative = false; inNumber = false; }
        }
        if (inNumber) result.Add(negative ? -value : value);
        return result;
    }
}
'@ | Set-Content -Path $harness -Encoding utf8

# Built against the machine's default framework references, not Unity's
# netstandard *reference* assemblies: those cannot be executed, only compiled
# against, and the harness has to actually run. Unity compatibility of the
# front end itself is covered by Invoke-LiteRtLmRuntimeCompileCheck.ps1.
$arguments = @("-nologo", "-target:exe", "-out:$exe", "-langversion:9.0", $source, $harness)

& $csc @arguments | Where-Object { $_ -match "error" } | ForEach-Object { Write-Host $_ -ForegroundColor Red }
if ($LASTEXITCODE -ne 0) { throw "harness compilation failed." }

& $exe $IndexerJson $ReferenceJson
exit $LASTEXITCODE
