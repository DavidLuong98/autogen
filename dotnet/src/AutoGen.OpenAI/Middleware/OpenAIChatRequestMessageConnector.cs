// Copyright (c) Microsoft Corporation. All rights reserved.
// OpenAIChatRequestMessageConnector.cs

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using OpenAI.Chat;

namespace AutoGen.OpenAI;

/// <summary>
/// This middleware converts the incoming <see cref="IMessage"/> to <see cref="IMessage{ChatRequestMessage}" /> where T is <see cref="ChatMessage"/> before sending to agent. And converts the output <see cref="ChatMessage"/> to <see cref="IMessage"/> after receiving from agent.
/// <para>Supported <see cref="IMessage"/> are</para>
/// <para>- <see cref="TextMessage"/></para> 
/// <para>- <see cref="ImageMessage"/></para> 
/// <para>- <see cref="MultiModalMessage"/></para>
/// <para>- <see cref="ToolCallMessage"/></para>
/// <para>- <see cref="ToolCallResultMessage"/></para>
/// <para>- <see cref="IMessage{ChatRequestMessage}"/> where T is <see cref="ChatMessage"/></para>
/// <para>- <see cref="AggregateMessage{TMessage1, TMessage2}"/> where TMessage1 is <see cref="ToolCallMessage"/> and TMessage2 is <see cref="ToolCallResultMessage"/></para>
/// </summary>
public class OpenAIChatRequestMessageConnector : IStreamingMiddleware
{
    private readonly bool strictMode;

    /// <summary>
    /// Create a new instance of <see cref="OpenAIChatRequestMessageConnector"/>.
    /// </summary>
    /// <param name="strictMode">If true, <see cref="OpenAIChatRequestMessageConnector"/> will throw an <see cref="InvalidOperationException"/>
    /// When the message type is not supported. If false, it will ignore the unsupported message type.</param>
    public OpenAIChatRequestMessageConnector(bool strictMode = false)
    {
        this.strictMode = strictMode;
    }

    public string? Name => nameof(OpenAIChatRequestMessageConnector);

    public async Task<IMessage> InvokeAsync(MiddlewareContext context, IAgent agent, CancellationToken cancellationToken = default)
    {
        var chatMessages = ProcessIncomingMessages(agent, context.Messages);

        var reply = await agent.GenerateReplyAsync(chatMessages, context.Options, cancellationToken);

        return PostProcessMessage(reply);
    }

    public async IAsyncEnumerable<IMessage> InvokeAsync(
        MiddlewareContext context,
        IStreamingAgent agent,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var chatMessages = ProcessIncomingMessages(agent, context.Messages);
        var streamingReply = agent.GenerateStreamingReplyAsync(chatMessages, context.Options, cancellationToken);
        string? currentToolName = null;
        await foreach (var reply in streamingReply)
        {
            if (reply is IMessage<StreamingChatCompletionUpdate> update)
            {
                if (update.Content.FunctionCallUpdate is not null)
                {
                    currentToolName = update.Content.FunctionCallUpdate.FunctionName;
                }
                else if (update.Content.ToolCallUpdates.Count > 0)
                {
                    currentToolName = update.Content.ToolCallUpdates[0].FunctionName;
                }

                var postProcessMessage = PostProcessStreamingMessage(update, currentToolName);
                if (postProcessMessage != null)
                {
                    yield return postProcessMessage;
                }
            }
            else
            {
                if (this.strictMode)
                {
                    throw new InvalidOperationException($"Invalid streaming message type {reply.GetType().Name}");
                }
                else
                {
                    yield return reply;
                }
            }
        }
    }

    private IMessage PostProcessMessage(IMessage message)
    {
        return message switch
        {
            IMessage<ChatCompletion> m => PostProcessChatResponseMessage(m.Content, m.From),
            _ when strictMode is false => message,
            _ => throw new InvalidOperationException($"Invalid return message type {message.GetType().Name}"),
        };
    }

    private IMessage? PostProcessStreamingMessage(IMessage<StreamingChatCompletionUpdate> update, string? currentToolName)
    {
        throw new NotImplementedException();
        /*
        if (update.Content.ContentUpdate is string contentUpdate)
        {
            // text message
            return new TextMessageUpdate(Role.Assistant, contentUpdate, from: update.From);
        }
        else if (update.Content.FunctionName is string functionName)
        {
            return new ToolCallMessageUpdate(functionName, string.Empty, from: update.From);
        }
        else if (update.Content.FunctionArgumentsUpdate is string functionArgumentsUpdate && currentToolName is string)
        {
            return new ToolCallMessageUpdate(currentToolName, functionArgumentsUpdate, from: update.From);
        }
        else if (update.Content.ToolCallUpdate is StreamingFunctionToolCallUpdate tooCallUpdate && currentToolName is string)
        {
            return new ToolCallMessageUpdate(tooCallUpdate.Name ?? currentToolName, tooCallUpdate.ArgumentsUpdate, from: update.From);
        }
        else
        {
            return null;
        }*/
    }

    private IMessage PostProcessChatResponseMessage(ChatCompletion chatCompletion, string? from)
    {
        var textContent = chatCompletion.Content[0].Text;
        switch (chatCompletion.FinishReason)
        {
            case ChatFinishReason.Stop:
                return new TextMessage(Role.Assistant, textContent, from);

            case ChatFinishReason.ToolCalls:
                return new ToolCallMessage(chatCompletion.ToolCalls.Select(tc =>
                    new ToolCall(tc.FunctionName, tc.FunctionArguments) { ToolCallId = tc.Id }), from)
                {
                    Content = textContent,
                };

            case ChatFinishReason.FunctionCall:
                var functionCall = chatCompletion.FunctionCall;
                return new ToolCallMessage(functionCall.FunctionName, functionCall.FunctionArguments, from)
                {
                    Content = textContent,
                };

            case ChatFinishReason.Length:
                throw new InvalidOperationException("The content that was submitted exceeded the length. Please shorten the input.");

            case ChatFinishReason.ContentFilter:
                throw new InvalidOperationException("The content is filtered because its potential risk. Please try another input.");
        }

        throw new InvalidOperationException("Invalid ChatResponseMessage");
    }

    public IEnumerable<IMessage> ProcessIncomingMessages(IAgent agent, IEnumerable<IMessage> messages)
    {
        return messages.SelectMany<IMessage, IMessage>(m =>
        {
            if (m is IMessage<ChatMessage> crm)
            {
                return [crm];
            }
            else
            {
                var chatRequestMessages = m switch
                {
                    TextMessage textMessage => ProcessTextMessage(agent, textMessage),
                    ImageMessage imageMessage when (imageMessage.From is null || imageMessage.From != agent.Name) => ProcessImageMessage(agent, imageMessage),
                    MultiModalMessage multiModalMessage when (multiModalMessage.From is null || multiModalMessage.From != agent.Name) => ProcessMultiModalMessage(agent, multiModalMessage),
                    ToolCallMessage toolCallMessage when (toolCallMessage.From is null || toolCallMessage.From == agent.Name) => ProcessToolCallMessage(agent, toolCallMessage),
                    ToolCallResultMessage toolCallResultMessage => ProcessToolCallResultMessage(toolCallResultMessage),
                    AggregateMessage<ToolCallMessage, ToolCallResultMessage> aggregateMessage => ProcessFunctionCallMiddlewareMessage(agent, aggregateMessage),
#pragma warning disable CS0618 // deprecated
                    Message msg => ProcessMessage(agent, msg),
#pragma warning restore CS0618 // deprecated
                    _ when strictMode is false => [],
                    _ => throw new InvalidOperationException($"Invalid message type: {m.GetType().Name}"),
                };

                if (chatRequestMessages.Any())
                {
                    return chatRequestMessages.Select(cm => MessageEnvelope.Create(cm, m.From));
                }
                else
                {
                    return [m];
                }
            }
        });
    }

    [Obsolete("This method is deprecated, please use ProcessIncomingMessages(IAgent agent, IEnumerable<IMessage> messages) instead.")]
    private IEnumerable<ChatMessage> ProcessIncomingMessagesForSelf(Message message)
    {
        if (message.Role == Role.System)
        {
            return new[] { new SystemChatMessage(message.Content) };
        }
        else if (message.Content is string content && content is { Length: > 0 })
        {
            if (message.FunctionName is null)
            {
                return new[] { new AssistantChatMessage(message.Content) };
            }
            else
            {
                return new[] { new ToolChatMessage(content, message.FunctionName) };
            }
        }
        else if (message.FunctionName is string functionName)
        {
            var msg = new AssistantChatMessage(content: null)
            {
                FunctionCall = new ChatFunctionCall(functionName, message.FunctionArguments)
            };

            return new[]
            {
                msg,
            };
        }
        else
        {
            throw new InvalidOperationException("Invalid Message as message from self.");
        }
    }

    [Obsolete("This method is deprecated, please use ProcessIncomingMessages(IAgent agent, IEnumerable<IMessage> messages) instead.")]
    private IEnumerable<ChatMessage> ProcessIncomingMessagesForOther(Message message)
    {
        if (message.Role == Role.System)
        {
            return [new SystemChatMessage(message.Content) { ParticipantName = message.From }];
        }
        else if (message.Content is { Length: > 0 } content)
        {
            if (message.FunctionName is not null)
            {
                return new[] { new ToolChatMessage(content, message.FunctionName) };
            }

            return [new UserChatMessage(message.Content) { ParticipantName = message.From }];
        }
        else if (message.FunctionName is string _)
        {
            return [new UserChatMessage("// Message type is not supported") { ParticipantName = message.From }];
        }
        else
        {
            throw new InvalidOperationException("Invalid Message as message from other.");
        }
    }

    private IEnumerable<ChatMessage> ProcessTextMessage(IAgent agent, TextMessage message)
    {
        if (message.Role == Role.System)
        {
            return [new SystemChatMessage(message.Content) { ParticipantName = message.From }];
        }

        if (agent.Name == message.From)
        {
            return [new AssistantChatMessage(message.Content) { ParticipantName = agent.Name }];
        }
        else
        {
            return message.From switch
            {
                null when message.Role == Role.User => [new UserChatMessage(message.Content)],
                null when message.Role == Role.Assistant => [new AssistantChatMessage(message.Content)],
                null => throw new InvalidOperationException("Invalid Role"),
                _ => [new UserChatMessage(message.Content) { ParticipantName = message.From }]
            };
        }
    }

    private IEnumerable<ChatMessage> ProcessImageMessage(IAgent agent, ImageMessage message)
    {
        if (agent.Name == message.From)
        {
            // image message from assistant is not supported
            throw new ArgumentException("ImageMessage is not supported when message.From is the same with agent");
        }

        var imageContentItem = this.CreateChatMessageImageContentItemFromImageMessage(message);
        return [new UserChatMessage([imageContentItem]) { ParticipantName = message.From }];
    }

    private IEnumerable<ChatMessage> ProcessMultiModalMessage(IAgent agent, MultiModalMessage message)
    {
        if (agent.Name == message.From)
        {
            // image message from assistant is not supported
            throw new ArgumentException("MultiModalMessage is not supported when message.From is the same with agent");
        }

        IEnumerable<ChatMessageContentPart> items = message.Content.Select<IMessage, ChatMessageContentPart>(ci => ci switch
        {
            TextMessage text => ChatMessageContentPart.CreateTextMessageContentPart(text.Content),
            ImageMessage image => this.CreateChatMessageImageContentItemFromImageMessage(image),
            _ => throw new NotImplementedException(),
        });

        return [new UserChatMessage(items) { ParticipantName = message.From }];
    }

    private ChatMessageContentPart CreateChatMessageImageContentItemFromImageMessage(ImageMessage message)
    {
        return message.Data is null && message.Url is not null
            ? ChatMessageContentPart.CreateImageMessageContentPart(new Uri(message.Url))
            : ChatMessageContentPart.CreateImageMessageContentPart(message.Data, message.Data?.MediaType);
    }

    private IEnumerable<ChatMessage> ProcessToolCallMessage(IAgent agent, ToolCallMessage message)
    {
        if (message.From is not null && message.From != agent.Name)
        {
            throw new ArgumentException("ToolCallMessage is not supported when message.From is not the same with agent");
        }

        var toolCall = message.ToolCalls.Select((tc, i) =>
            ChatToolCall.CreateFunctionToolCall(tc.ToolCallId ?? $"{tc.FunctionName}_{i}", tc.FunctionName,
                tc.FunctionArguments));

        var textContent = message.GetContent() ?? string.Empty;
        var chatRequestMessage = new AssistantChatMessage(textContent) { ParticipantName = message.From };
        foreach (var tc in toolCall)
        {
            chatRequestMessage.ToolCalls.Add(tc);
        }

        return [chatRequestMessage];
    }

    private IEnumerable<ChatMessage> ProcessToolCallResultMessage(ToolCallResultMessage message)
    {
        return message.ToolCalls
            .Where(tc => tc.Result is not null)
            .Select((tc, i) => new ToolChatMessage(tc.Result, tc.ToolCallId ?? $"{tc.FunctionName}_{i}"));
    }

    [Obsolete("This method is deprecated, please use ProcessIncomingMessages(IAgent agent, IEnumerable<IMessage> messages) instead.")]
    private IEnumerable<ChatMessage> ProcessMessage(IAgent agent, Message message)
    {
        if (message.From is not null && message.From != agent.Name)
        {
            return ProcessIncomingMessagesForOther(message);
        }
        else
        {
            return ProcessIncomingMessagesForSelf(message);
        }
    }

    private IEnumerable<ChatMessage> ProcessFunctionCallMiddlewareMessage(IAgent agent, AggregateMessage<ToolCallMessage, ToolCallResultMessage> aggregateMessage)
    {
        if (aggregateMessage.From is not null && aggregateMessage.From != agent.Name)
        {
            // convert as user message
            var resultMessage = aggregateMessage.Message2;

            return resultMessage.ToolCalls.Select(tc => new UserChatMessage(tc.Result) { ParticipantName = aggregateMessage.From });
        }
        else
        {
            var toolCallMessage1 = aggregateMessage.Message1;
            var toolCallResultMessage = aggregateMessage.Message2;

            var assistantMessage = this.ProcessToolCallMessage(agent, toolCallMessage1);
            var toolCallResults = this.ProcessToolCallResultMessage(toolCallResultMessage);

            return assistantMessage.Concat(toolCallResults);
        }
    }
}
