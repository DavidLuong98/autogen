// Copyright (c) Microsoft Corporation. All rights reserved.
// OpenAIChatAgent.cs

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading;
using System.Threading.Tasks;
using AutoGen.OpenAI.Extension;
using OpenAI;
using OpenAI.Chat;

namespace AutoGen.OpenAI;

/// <summary>
/// OpenAI client agent. This agent is a thin wrapper around <see cref="OpenAIClient"/> to provide a simple interface for chat completions.
/// To better work with other agents, it's recommended to use <see cref="GPTAgent"/> which supports more message types and have a better compatibility with other agents.
/// <para><see cref="OpenAIChatAgent" /> supports the following message types:</para>
/// <list type="bullet">
/// <item>
/// <see cref="MessageEnvelope{T}"/> where T is <see cref="ChatMessage"/>: chat request message.
/// </item>
/// </list>
/// <para><see cref="OpenAIChatAgent" /> returns the following message types:</para>
/// <list type="bullet">
/// <item>
/// <see cref="MessageEnvelope{T}"/> where T is <see cref="ChatMessage"/>: chat response message.
/// <see cref="MessageEnvelope{T}"/> where T is <see cref="StreamingChatCompletionUpdate"/>: streaming chat completions update.
/// </item>
/// </list>
/// </summary>
public class OpenAIChatAgent : IStreamingAgent
{
    private readonly OpenAIClient openAIClient;
    private readonly ChatCompletionOptions options;
    private readonly string systemMessage;

    /// <summary>
    /// Create a new instance of <see cref="OpenAIChatAgent"/>.
    /// </summary>
    /// <param name="openAIClient">openai client</param>
    /// <param name="name">agent name</param>
    /// <param name="modelName">model name. e.g. gpt-turbo-3.5</param>
    /// <param name="systemMessage">system message</param>
    /// <param name="temperature">temperature</param>
    /// <param name="maxTokens">max tokens to generated</param>
    /// <param name="chatResponseFormat">response format, set it to <see cref="JsonObject"/> to enable json mode.</param>
    /// <param name="seed">seed to use, set it to enable deterministic output</param>
    /// <param name="functions">functions</param>
    public OpenAIChatAgent(
        OpenAIClient openAIClient,
        string name,
        string modelName,
        string systemMessage = "You are a helpful AI assistant",
        float temperature = 0.7f,
        int maxTokens = 1024,
        int? seed = null,
        ChatResponseFormat? chatResponseFormat = null,
        IEnumerable<ChatTool>? functions = null)
        : this(
            openAIClient: openAIClient,
            name: name,
            options: CreateChatCompletionOptions(modelName, temperature, maxTokens, seed, chatResponseFormat, functions),
            systemMessage: systemMessage)
    {
    }

    /// <summary>
    /// Create a new instance of <see cref="OpenAIChatAgent"/>.
    /// </summary>
    /// <param name="openAIClient">openai client</param>
    /// <param name="name">agent name</param>
    /// <param name="systemMessage">system message</param>
    /// <param name="options">chat completion option. The option can't contain messages</param>
    public OpenAIChatAgent(
        OpenAIClient openAIClient,
        string name,
        ChatCompletionOptions options,
        string systemMessage = "You are a helpful AI assistant")
    {
        this.openAIClient = openAIClient;
        this.Name = name;
        this.options = options;
        this.systemMessage = systemMessage;
    }

    public string Name { get; }

    public async Task<IMessage> GenerateReplyAsync(
        IEnumerable<IMessage> messages,
        GenerateReplyOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        var chatCompletionOptions = this.CreateChatCompletionOptions(options);

        string model = string.Empty;
        if (string.IsNullOrEmpty(model))
        {
            throw new NotImplementedException();
        }

        var reply = await this.openAIClient.GetChatClient(model)
            .CompleteChatAsync(ConstructChatMessages(messages), chatCompletionOptions, cancellationToken);

        return new MessageEnvelope<ChatCompletion>(reply.Value, from: this.Name);
    }

    public async IAsyncEnumerable<IMessage> GenerateStreamingReplyAsync(
        IEnumerable<IMessage> messages,
        GenerateReplyOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var chatCompletionOptions = this.CreateChatCompletionOptions(options);
        var response = this.openAIClient.GetChatClient(model: string.Empty)
            .CompleteChatStreamingAsync(ConstructChatMessages(messages), chatCompletionOptions, cancellationToken);

        await foreach (var update in response)
        {
            yield return new MessageEnvelope<StreamingChatCompletionUpdate>(update, from: this.Name);
        }
    }

    private IEnumerable<ChatMessage> ConstructChatMessages(IEnumerable<IMessage> messages)
    {
        var chatMessages = messages.Select(m => m switch
        {
            IMessage<ChatMessage> chatRequestMessage => chatRequestMessage.Content,
            _ => throw new ArgumentException("Invalid message type")
        });

        // add system message if there's no system message in messages
        if (!chatMessages.Any(m => m is SystemChatMessage))
        {
            chatMessages = new[] { new SystemChatMessage(systemMessage) }.Concat(chatMessages);
        }

        return chatMessages;
    }

    private ChatCompletionOptions CreateChatCompletionOptions(GenerateReplyOptions? options)
    {
        // clone the options by serializing and deserializing
        var json = JsonSerializer.Serialize(this.options);
        var settings = JsonSerializer.Deserialize<ChatCompletionOptions>(json) ??
                       throw new InvalidOperationException("Failed to clone options");

        // settings.Temperature = options?.Temperature ?? settings.Temperature;
        // settings.MaxTokens = options?.MaxToken ?? settings.MaxTokens;

        foreach (var functions in this.options.Tools)
        {
            settings.Tools.Add(functions);
        }

        foreach (var stopSequence in this.options.StopSequences)
        {
            settings.StopSequences.Add(stopSequence);
        }

        var openAiFunctionDefinitions = options?.Functions?.Select(f => f.ToOpenAIFunctionDefinition()).ToList();
        if (openAiFunctionDefinitions is { Count: > 0 })
        {
            foreach (var f in openAiFunctionDefinitions)
            {
                settings.Tools.Add(ChatTool.CreateFunctionTool(f.FunctionName, f.Description, f.Parameters));
            }
        }

        if (options?.StopSequence is var sequence && sequence is { Length: > 0 })
        {
            foreach (var seq in sequence)
            {
                settings.StopSequences.Add(seq);
            }
        }

        return settings;
    }

    private static ChatCompletionOptions CreateChatCompletionOptions(
        string modelName,
        float temperature = 0.7f,
        int maxTokens = 1024,
        int? seed = null,
        ChatResponseFormat? responseFormat = null,
        IEnumerable<ChatTool>? chatTools = null)
    {
        var options = new ChatCompletionOptions
        {
            Temperature = temperature,
            MaxTokens = maxTokens,
            Seed = seed,
            ResponseFormat = responseFormat,
        };

        if (chatTools is not null)
        {
            foreach (var f in chatTools)
            {
                options.Tools.Add(f);
            }
        }

        return options;
    }
}
