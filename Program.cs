using DevAgent.Agents;
using DevAgent.Memory;
using DevAgent.Plugins;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Agents;
using Microsoft.SemanticKernel.ChatCompletion;

// ── Configuration ────────────────────────────────────────────────────────────
string apiKey = Environment.GetEnvironmentVariable("OPENAI_API_KEY")
    ?? throw new Exception("Set the OPENAI_API_KEY environment variable.");

const string Model       = "gpt-4o-mini";
const string HistoryFile = "history.json";
const int    MaxHistory  = 40;

// ── Kernel setup ─────────────────────────────────────────────────────────────
// CodeAgent kernel: full access to all dev tools
var codeKernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(Model, apiKey)
    .Build();
codeKernel.Plugins.AddFromObject(new DevToolsPlugin(), "DevTools");

// ArchitectAgent kernel: read-only tools (no write/run/git)
var architectKernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(Model, apiKey)
    .Build();
architectKernel.Plugins.AddFromObject(new ReadOnlyDevToolsPlugin(), "DevTools");

// Router kernel: no tools, just classification
var routerKernel = Kernel.CreateBuilder()
    .AddOpenAIChatCompletion(Model, apiKey)
    .Build();

// ── Agent + router setup ──────────────────────────────────────────────────────
var codeAgent      = DevAgentFactory.CreateCodeAgent(codeKernel);
var architectAgent = DevAgentFactory.CreateArchitectAgent(architectKernel);
var router         = new Router(routerKernel);

// ── Load conversation history ─────────────────────────────────────────────────
var history = HistoryManager.Load(HistoryFile, MaxHistory);
if (history.Count > 0)
{
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($"📂 Loaded {history.Count} messages from previous session.\n");
    Console.ResetColor();
}

// ── Main loop ─────────────────────────────────────────────────────────────────
Console.ForegroundColor = ConsoleColor.Cyan;
Console.WriteLine("🤖 Dev Agent ready! Type your task (or 'exit' to quit, 'clear' to reset history).\n");
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
        history.Clear();
        File.Delete(HistoryFile);
        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine("🗑️  History cleared.\n");
        Console.ResetColor();
        continue;
    }

    history.AddUserMessage(userInput);

    // Route the request
    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.Write("🔀 Routing...");
    Console.ResetColor();

    var route = await router.RouteAsync(userInput);
    var (agent, agentLabel) = route == "code"
        ? (codeAgent, "🔨 CodeAgent")
        : (architectAgent, "💬 ArchitectAgent");

    Console.ForegroundColor = ConsoleColor.DarkGray;
    Console.WriteLine($" → {agentLabel}");
    Console.ResetColor();

    // Stream the agent response
    Console.ForegroundColor = ConsoleColor.Cyan;
    Console.Write($"\n{agentLabel}: ");
    Console.ResetColor();

    var responseBuilder = new System.Text.StringBuilder();

    await foreach (var chunk in agent.InvokeStreamingAsync(history))
    {
        var text = chunk.ToString();
        if (!string.IsNullOrEmpty(text))
        {
            Console.Write(text);
            responseBuilder.Append(text);
        }
    }

    Console.WriteLine("\n");

    // Add final response to history and save
    history.AddAssistantMessage(responseBuilder.ToString());
    HistoryManager.Save(history, HistoryFile, MaxHistory);
}
