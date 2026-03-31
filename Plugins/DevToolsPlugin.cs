using System.ComponentModel;
using Microsoft.SemanticKernel;

namespace DevAgent.Plugins;

public class DevToolsPlugin
{
    [KernelFunction, Description("Reads the content of a file at the given path.")]
    public string ReadFile([Description("Absolute or relative file path")] string path)
    {
        return File.Exists(path) ? File.ReadAllText(path) : $"File not found: {path}";
    }

    [KernelFunction, Description("Writes (or overwrites) a file with the given content.")]
    public string WriteFile(
        [Description("File path to write")] string path,
        [Description("Content to write")] string content)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
        File.WriteAllText(path, content);
        return $"File written: {path}";
    }

    [KernelFunction, Description("Runs a shell command and returns stdout + stderr.")]
    public string RunCommand([Description("The command to execute")] string command)
    {
        try
        {
            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = $"/c {command}",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                }
            };
            process.Start();
            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();
            process.WaitForExit();
            return string.IsNullOrEmpty(error) ? output : $"{output}\nSTDERR: {error}";
        }
        catch (Exception ex)
        {
            return $"Error running command: {ex.Message}";
        }
    }

    [KernelFunction, Description("Lists files and directories at the given path.")]
    public string ListFiles([Description("Directory path")] string path)
    {
        if (!Directory.Exists(path)) return $"Directory not found: {path}";
        return string.Join("\n", Directory.GetFileSystemEntries(path));
    }

    [KernelFunction, Description("Searches for a text pattern inside files within a directory. Returns matching lines with file paths and line numbers.")]
    public string SearchFiles(
        [Description("Directory to search in")] string directory,
        [Description("Text or regex pattern to search for")] string pattern,
        [Description("Optional file extension filter, e.g. '.cs' or '.ts'")] string? extension = null)
    {
        if (!Directory.Exists(directory))
            return $"Directory not found: {directory}";

        var searchPattern = string.IsNullOrEmpty(extension) ? "*.*" : $"*{extension}";
        var files = Directory.GetFiles(directory, searchPattern, SearchOption.AllDirectories);

        var results = new List<string>();
        foreach (var file in files)
        {
            var lines = File.ReadAllLines(file);
            for (int i = 0; i < lines.Length; i++)
            {
                if (lines[i].Contains(pattern, StringComparison.OrdinalIgnoreCase))
                    results.Add($"{file}:{i + 1}: {lines[i].Trim()}");
            }
        }

        if (results.Count == 0) return $"No matches found for '{pattern}'.";
        if (results.Count > 100) results = [..results.Take(100), $"... ({results.Count - 100} more results truncated)"];

        return string.Join("\n", results);
    }

    [KernelFunction, Description("Returns the git status of a repository, showing changed, staged, and untracked files.")]
    public string GitStatus([Description("Path to the git repository root")] string repoPath)
        => RunGit(repoPath, "status");

    [KernelFunction, Description("Returns the git diff of a repository. Shows unstaged changes by default, or staged changes if specified.")]
    public string GitDiff(
        [Description("Path to the git repository root")] string repoPath,
        [Description("If true, shows staged (--cached) diff. Default false.")] bool staged = false,
        [Description("Optional: limit diff to a specific file")] string? filePath = null)
    {
        var arguments = staged ? "diff --cached" : "diff";
        if (!string.IsNullOrEmpty(filePath)) arguments += $" -- \"{filePath}\"";
        return RunGit(repoPath, arguments);
    }

    [KernelFunction, Description("Stages specified files (or all changes) and creates a git commit with the given message.")]
    public string GitCommit(
        [Description("Path to the git repository root")] string repoPath,
        [Description("Commit message")] string message,
        [Description("Files to stage. If empty, stages all changes (git add .).")] string[]? files = null)
    {
        if (files is { Length: > 0 })
            RunGit(repoPath, $"add {string.Join(" ", files.Select(f => $"\"{f}\""))}");
        else
            RunGit(repoPath, "add .");

        return RunGit(repoPath, $"commit -m \"{message}\"");
    }

    private static string RunGit(string repoPath, string arguments)
    {
        if (!Directory.Exists(repoPath))
            return $"Error: Directory not found: {repoPath}";

        try
        {
            var process = new System.Diagnostics.Process
            {
                StartInfo = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = "git",
                    Arguments = arguments,
                    WorkingDirectory = repoPath,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                }
            };
            process.Start();
            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();
            process.WaitForExit();
            return string.IsNullOrWhiteSpace(output) ? error.Trim() : output.Trim();
        }
        catch (Exception ex)
        {
            return $"Error running git: {ex.Message}";
        }
    }
}
