using OpenAI.Chat;
using System.Text.Json;

// ── Configuration ──────────────────────────────────────────────────────────
string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
    ?? throw new Exception("Set the OPENAI_API_KEY environment variable.");

var client = new ChatClient("gpt-4o-mini", apiKey);

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

// ── System prompt ───────────────────────────────────────────────────────────
var messages = new List<ChatMessage>
{
    new SystemChatMessage("""
        You are a helpful developer assistant agent. You help the user with software development tasks.
        You have tools to read files, write files, run shell commands, and list directories.
        Use these tools to fulfill the user's requests. Always reason step by step before acting.
        After completing a task, summarize what you did.
        """)
};

// ── ReAct loop ───────────────────────────────────────────────────────────────
Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("🤖 Dev Agent ready! Type your task (or 'exit' to quit).\n");
Console.ResetColor();

while (true)
{
    Console.ForegroundColor = ConsoleColor.Green;
    Console.Write("You: ");
    Console.ResetColor();

    var userInput = Console.ReadLine();
    if (string.IsNullOrWhiteSpace(userInput)) continue;
    if (userInput.Equals("exit", StringComparison.OrdinalIgnoreCase)) break;

    messages.Add(new UserChatMessage(userInput));

    // Agent loop — keeps running until the model stops calling tools
    while (true)
    {
        var response = await client.CompleteChatAsync(messages, new ChatCompletionOptions
        {
            Tools = { tools[0], tools[1], tools[2], tools[3], tools[4] }
        });

        var result = response.Value;

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
            break;
        }
    }
}
