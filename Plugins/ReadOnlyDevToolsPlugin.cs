using System.ComponentModel;
using Microsoft.SemanticKernel;

namespace DevAgent.Plugins;

// Exposes only read-only tools for the ArchitectAgent
public class ReadOnlyDevToolsPlugin
{
    private readonly DevToolsPlugin _tools = new();

    [KernelFunction, Description("Reads the content of a file at the given path.")]
    public string ReadFile([Description("Absolute or relative file path")] string path)
        => _tools.ReadFile(path);

    [KernelFunction, Description("Lists files and directories at the given path.")]
    public string ListFiles([Description("Directory path")] string path)
        => _tools.ListFiles(path);

    [KernelFunction, Description("Searches for a text pattern inside files within a directory. Returns matching lines with file paths and line numbers.")]
    public string SearchFiles(
        [Description("Directory to search in")] string directory,
        [Description("Text or regex pattern to search for")] string pattern,
        [Description("Optional file extension filter, e.g. '.cs' or '.ts'")] string? extension = null)
        => _tools.SearchFiles(directory, pattern, extension);
}
