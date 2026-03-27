using OpenAI.Chat;
using System.Text.Json;

// ── Configuration ──────────────────────────────────────────────────────────
string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
    ?? throw new Exception("Set the OPENAI_API_KEY environment variable.");

var client = new ChatClient("gpt-4o-mini", apiKey);

const string HistoryFile = "history.json";
const int    MaxHistory  = 40; // max messages to keep (excluding system prompt)

// gpt-4o-mini pricing per 1M tokens (as of early 2025)
const double CostPerInputToken  = 0.15 / 1_000_000;
const double CostPerOutputToken = 0.60 / 1_000_000;

int sessionInputTokens  = 0;
int sessionOutputTokens = 0;

// ── Tool definitions ────────────────────────────────────────────────────────
var tools = new List<ChatTool>
{
    ChatTool.CreateFunctionTool(
        "read_file",
        "Reads the content of a file at the given path.",
        BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Absolute or relative file path" }
            },
            "required": ["path"]
        }
        """)),

    ChatTool.CreateFunctionTool(
        "write_file",
        "Writes (or overwrites) a file with the given content.",
        BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "path":    { "type": "string", "description": "File path to write" },
                "content": { "type": "string", "description": "Content to write" }
            },
            "required": ["path", "content"]
        }
        """)),

    ChatTool.CreateFunctionTool(
        "run_command",
        "Runs a shell command and returns stdout + stderr.",
        BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "The command to execute" }
            },
            "required": ["command"]
        }
        """)),

    ChatTool.CreateFunctionTool(
        "list_files",
        "Lists files in a directory.",
        BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path" }
            },
            "required": ["path"]
        }
        """)),

    ChatTool.CreateFunctionTool(
        "search_files",
        "Searches for a text pattern inside files within a directory. Returns matching lines with file paths and line numbers.",
        BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "directory": { "type": "string", "description": "Directory to search in" },
                "pattern":   { "type": "string", "description": "Text or regex pattern to search for" },
                "extension": { "type": "string", "description": "Optional file extension filter, e.g. '.cs' or '.ts'" }
            },
            "required": ["directory", "pattern"]
        }
        """)),

    ChatTool.CreateFunctionTool(
        "git_status",
        "Returns the git status of a repository, showing changed, staged, and untracked files.",
        BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "repo_path": { "type": "string", "description": "Path to the git repository root" }
            },
            "required": ["repo_path"]
        }
        """)),

    ChatTool.CreateFunctionTool(
        "git_diff",
        "Returns the git diff of a repository. Shows unstaged changes by default, or staged changes if specified.",
        BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "repo_path": { "type": "string", "description": "Path to the git repository root" },
                "staged":    { "type": "boolean", "description": "If true, shows staged (--cached) diff. Default false." },
                "file_path": { "type": "string", "description": "Optional: limit diff to a specific file" }
            },
            "required": ["repo_path"]
        }
        """)),

    ChatTool.CreateFunctionTool(
        "git_commit",
        "Stages specified files (or all changes) and creates a git commit with the given message.",
        BinaryData.FromString("""
        {
            "type": "object",
            "properties": {
                "repo_path": { "type": "string", "description": "Path to the git repository root" },
                "message":   { "type": "string", "description": "Commit message" },
                "files":     { "type": "array", "items": { "type": "string" }, "description": "Files to stage. If empty, stages all changes (git add .)." }
            },
            "required": ["repo_path", "message"]
        }
        """)),
};

// ── Tool execution ──────────────────────────────────────────────────────────
static string ExecuteTool(string name, string argsJson)
{
    var args = JsonDocument.Parse(argsJson).RootElement;

    return name switch
    {
        "read_file" => ExecuteReadFile(args),
        "write_file" => ExecuteWriteFile(args),
        "run_command" => ExecuteRunCommand(args),
        "list_files" => ExecuteListFiles(args),
        "search_files" => ExecuteSearchFiles(args),
        "git_status"   => ExecuteGitStatus(args),
        "git_diff"     => ExecuteGitDiff(args),
        "git_commit"   => ExecuteGitCommit(args),
        _ => $"Unknown tool: {name}"
    };
}

static string ExecuteReadFile(JsonElement args)
{
    var path = args.GetProperty("path").GetString()!;
    return File.Exists(path) ? File.ReadAllText(path) : $"File not found: {path}";
}

static string ExecuteWriteFile(JsonElement args)
{
    var path = args.GetProperty("path").GetString()!;
    var content = args.GetProperty("content").GetString()!;
    Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
    File.WriteAllText(path, content);
    return $"File written: {path}";
}

static string ExecuteRunCommand(JsonElement args)
{
    var command = args.GetProperty("command").GetString()!;
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

static string ExecuteListFiles(JsonElement args)
{
    var path = args.GetProperty("path").GetString()!;
    if (!Directory.Exists(path)) return $"Directory not found: {path}";
    var entries = Directory.GetFileSystemEntries(path);
    return string.Join("\n", entries);
}

static string ExecuteSearchFiles(JsonElement args)
{
    var directory = args.GetProperty("directory").GetString()!;
    var pattern   = args.GetProperty("pattern").GetString()!;
    var extension = args.TryGetProperty("extension", out var ext) ? ext.GetString() : null;

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

static string RunGit(string repoPath, string arguments)
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
        string error  = process.StandardError.ReadToEnd();
        process.WaitForExit();
        return string.IsNullOrWhiteSpace(output) ? error.Trim() : output.Trim();
    }
    catch (Exception ex)
    {
        return $"Error running git: {ex.Message}";
    }
}

static string ExecuteGitStatus(JsonElement args)
{
    var repoPath = args.GetProperty("repo_path").GetString()!;
    return RunGit(repoPath, "status");
}

static string ExecuteGitDiff(JsonElement args)
{
    var repoPath = args.GetProperty("repo_path").GetString()!;
    var staged   = args.TryGetProperty("staged", out var s) && s.GetBoolean();
    var filePath = args.TryGetProperty("file_path", out var f) ? f.GetString() : null;

    var arguments = staged ? "diff --cached" : "diff";
    if (!string.IsNullOrEmpty(filePath)) arguments += $" -- \"{filePath}\"";

    return RunGit(repoPath, arguments);
}

static string ExecuteGitCommit(JsonElement args)
{
    var repoPath = args.GetProperty("repo_path").GetString()!;
    var message  = args.GetProperty("message").GetString()!;
    var hasFiles = args.TryGetProperty("files", out var filesEl) && filesEl.GetArrayLength() > 0;

    if (hasFiles)
    {
        var files = filesEl.EnumerateArray().Select(f => $"\"{f.GetString()}\"");
        RunGit(repoPath, $"add {string.Join(" ", files)}");
    }
    else
    {
        RunGit(repoPath, "add .");
    }

    return RunGit(repoPath, $"commit -m \"{message}\"");
}

// ── Memory helpers ───────────────────────────────────────────────────────────

// Each entry stored as: { "role": "user"|"assistant", "content": "..." }
static void SaveHistory(IEnumerable<ChatMessage> messages, string path)
{
    var entries = messages
        .Where(m => m is UserChatMessage or AssistantChatMessage)
        .Select(m => new
        {
            role    = m is UserChatMessage ? "user" : "assistant",
            content = m is UserChatMessage u
                        ? u.Content[0].Text
                        : ((AssistantChatMessage)m).Content.Count > 0
                            ? ((AssistantChatMessage)m).Content[0].Text
                            : null
        })
        .Where(e => e.content != null)
        .ToList();

    File.WriteAllText(path, JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true }));
}

static List<ChatMessage> LoadHistory(string path)
{
    if (!File.Exists(path)) return [];

    var json    = File.ReadAllText(path);
    var entries = JsonSerializer.Deserialize<List<JsonElement>>(json) ?? [];
    var result  = new List<ChatMessage>();

    foreach (var entry in entries)
    {
        var role    = entry.GetProperty("role").GetString();
        var content = entry.GetProperty("content").GetString() ?? "";
        if (role == "user")      result.Add(new UserChatMessage(content));
        if (role == "assistant") result.Add(new AssistantChatMessage(content));
    }

    return result;
}

// ── System prompt ───────────────────────────────────────────────────────────
var messages = new List<ChatMessage>
{
    new SystemChatMessage("""
        You are an expert software developer assistant with deep knowledge of software engineering,
        clean code principles, and common development workflows.

        ## Your personality
        - You think like a senior developer: pragmatic, precise, and focused on working solutions.
        - You are direct and concise. You don't over-explain unless asked.
        - You care about code quality: you notice bad patterns, potential bugs, and improvements.

        ## How you work
        - Always reason step by step before taking action.
        - Before writing or modifying files, read them first so you understand the context.
        - When exploring an unfamiliar codebase, start with list_files to understand the structure,
          then read key files before drawing conclusions.
        - Prefer small, focused changes over large rewrites.
        - If a task is ambiguous, ask one clarifying question before proceeding.

        ## Your tools
        - read_file: Read the contents of a file.
        - write_file: Write or overwrite a file. Use with care — always read first.
        - run_command: Run a shell command (build, test, install packages, etc.).
        - list_files: List contents of a directory.
        - search_files: Search for a text pattern across files — use this to find usages, definitions, or references.
        - git_status: Show what files have changed in a git repo.
        - git_diff: Show the actual code changes (unstaged or staged).
        - git_commit: Stage and commit files with a message.

        ## Output style
        - Keep responses short and developer-friendly.
        - When showing code, always use code blocks with the correct language tag.
        - After completing a task, give a brief one or two sentence summary of what you did.
        - If you encounter an error, explain what went wrong and suggest a fix.
        """)
};

// Load previous conversation history
var history = LoadHistory(HistoryFile);
if (history.Count > 0)
{
    // Trim to max before loading so we don't start with a bloated context
    var trimmed = history.TakeLast(MaxHistory).ToList();
    messages.AddRange(trimmed);
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"📂 Loaded {trimmed.Count} messages from previous session.\n");
    Console.ResetColor();
}

// ── ReAct loop ───────────────────────────────────────────────────────────────
Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("🤖 Dev Agent ready! Type your task (or 'exit' to quit, 'clear' to reset history, 'stats' for token usage).\n");
Console.ResetColor();

while (true)
{
    Console.ForegroundColor = ConsoleColor.Green;
    Console.Write("You: ");
    Console.ResetColor();

    var userInput = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(userInput)) continue;
    if (userInput.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;
    if (userInput.Equals("clear", StringComparison.OrdinalIgnoreCase))
    {
        messages.RemoveAll(m => m is UserChatMessage or AssistantChatMessage);
        File.Delete(HistoryFile);
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("🗑️  History cleared.\n");
        Console.ResetColor();
        continue;
    }
    if (userInput.Equals("stats", StringComparison.OrdinalIgnoreCase))
    {
        var cost = (sessionInputTokens * CostPerInputToken) + (sessionOutputTokens * CostPerOutputToken);
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"📊 Session tokens: {sessionInputTokens} in / {sessionOutputTokens} out | Est. cost: ${cost:F5}\n");
        Console.ResetColor();
        continue;
    }

    messages.Add(new UserChatMessage(userInput));

    // Agent loop — keeps running until the model stops calling tools
    while (true)
    {
        var response = await client.CompleteChatAsync(messages, new ChatCompletionOptions
        {
            Tools = { tools[0], tools[1], tools[2], tools[3], tools[4], tools[5], tools[6], tools[7] }
        });

        var result = response.Value;

        // Accumulate token usage from every API call in the agent loop
        sessionInputTokens  += result.Usage?.InputTokenCount  ?? 0;
        sessionOutputTokens += result.Usage?.OutputTokenCount ?? 0;

        if (result.FinishReason == ChatFinishReason.ToolCalls)
        {
            // Add assistant message with tool calls
            messages.Add(new AssistantChatMessage(result));

            // Execute each tool and feed results back
            foreach (var toolCall in result.ToolCalls)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine($"\n⚙️  Tool: {toolCall.FunctionName}({toolCall.FunctionArguments})");
                Console.ResetColor();

                var toolResult = ExecuteTool(toolCall.FunctionName, toolCall.FunctionArguments.ToString());

                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.WriteLine($"   → {toolResult[..Math.Min(200, toolResult.Length)]}...");
                Console.ResetColor();

                messages.Add(new ToolChatMessage(toolCall.Id, toolResult));
            }
            // Continue the loop so the model can react to tool results
        }
        else
        {
            // Model is done — print final response
            var reply = result.Content[0].Text;
            messages.Add(new AssistantChatMessage(reply));

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine($"\n🤖 Agent: {reply}\n");
            Console.ResetColor();

            // Show token usage for this exchange
            var turnInput    = result.Usage?.InputTokenCount  ?? 0;
            var turnOutput   = result.Usage?.OutputTokenCount ?? 0;
            var sessionCost  = (sessionInputTokens * CostPerInputToken) + (sessionOutputTokens * CostPerOutputToken);
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine($"📊 Tokens: {turnInput} in / {turnOutput} out | Session total: {sessionInputTokens} in / {sessionOutputTokens} out | Est. cost: ${sessionCost:F5}\n");
            Console.ResetColor();

            // Trim and save history after every reply
            var toSave = messages.Where(m => m is UserChatMessage or AssistantChatMessage).TakeLast(MaxHistory);
            SaveHistory(toSave, HistoryFile);

            break;
        }
    }
}
