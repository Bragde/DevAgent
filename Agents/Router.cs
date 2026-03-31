using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

namespace DevAgent.Agents;

public class Router(Kernel kernel)
{
    private const string RouterInstructions = """
        You are a request router for a developer assistant. Classify the user's request as exactly one word:
        - "code" — the request involves doing something: writing, editing, running, reading, or inspecting code, files, commands, or git operations.
        - "chat" — the request involves understanding something: explaining concepts, discussing architecture, reviewing tradeoffs, or answering technical questions.
        Respond with only "code" or "chat". Nothing else.
        """;

    public async Task<string> RouteAsync(string userInput)
    {
        var chat = kernel.GetRequiredService<IChatCompletionService>();

        var messages = new ChatHistory(RouterInstructions);
        messages.AddUserMessage(userInput);

        var result = await chat.GetChatMessageContentAsync(messages);
        var classification = result.Content?.Trim().ToLower() ?? "code";

        return classification == "chat" ? "chat" : "code";
    }
}
