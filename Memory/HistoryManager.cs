using System.Text.Json;
using Microsoft.SemanticKernel.ChatCompletion;

namespace DevAgent.Memory;

public static class HistoryManager
{
    // Each entry stored as: { "role": "user"|"assistant", "content": "..." }
    public static void Save(ChatHistory history, string path, int maxMessages = 40)
    {
        var entries = history
            .Where(m => m.Role == AuthorRole.User || m.Role == AuthorRole.Assistant)
            .TakeLast(maxMessages)
            .Select(m => new { role = m.Role.Label, content = m.Content })
            .ToList();

        File.WriteAllText(path, JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static ChatHistory Load(string path, int maxMessages = 40)
    {
        var history = new ChatHistory();
        if (!File.Exists(path)) return history;

        var json    = File.ReadAllText(path);
        var entries = JsonSerializer.Deserialize<List<JsonElement>>(json) ?? [];

        foreach (var entry in entries.TakeLast(maxMessages))
        {
            var role    = entry.GetProperty("role").GetString();
            var content = entry.GetProperty("content").GetString() ?? "";
            if (role == "user")      history.AddUserMessage(content);
            if (role == "assistant") history.AddAssistantMessage(content);
        }

        return history;
    }
}
