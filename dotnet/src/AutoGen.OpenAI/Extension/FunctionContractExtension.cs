﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// FunctionContractExtension.cs

using System;
using System.Collections.Generic;
using Json.Schema;
using Json.Schema.Generation;
using OpenAI.Assistants;

namespace AutoGen.OpenAI.Extension;

public static class FunctionContractExtension
{
    /// <summary>
    /// Convert a <see cref="FunctionContract"/> to a <see cref="FunctionToolDefinition"/> that can be used in gpt funciton call.
    /// </summary>
    /// <param name="functionContract">function contract</param>
    /// <returns><see cref="FunctionToolDefinition"/></returns>
    public static FunctionToolDefinition ToOpenAIFunctionDefinition(this FunctionContract functionContract)
    {
        var requiredParameterNames = new List<string>();
        var propertiesSchemas = new Dictionary<string, JsonSchema>();
        var propertySchemaBuilder = new JsonSchemaBuilder().Type(SchemaValueType.Object);
        foreach (var param in functionContract.Parameters ?? [])
        {
            if (param.Name is null)
            {
                throw new InvalidOperationException("Parameter name cannot be null");
            }

            var schemaBuilder = new JsonSchemaBuilder().FromType(param.ParameterType ?? throw new ArgumentNullException(nameof(param.ParameterType)));
            if (param.Description != null)
            {
                schemaBuilder = schemaBuilder.Description(param.Description);
            }

            if (param.IsRequired)
            {
                requiredParameterNames.Add(param.Name);
            }

            var schema = schemaBuilder.Build();
            propertiesSchemas[param.Name] = schema;

        }
        propertySchemaBuilder = propertySchemaBuilder.Properties(propertiesSchemas);
        propertySchemaBuilder = propertySchemaBuilder.Required(requiredParameterNames);

        var option = new System.Text.Json.JsonSerializerOptions()
        {
            PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase
        };

        var functionDefinition = new FunctionToolDefinition
        {
            FunctionName = functionContract.Name,
            Description = functionContract.Description,
            Parameters = BinaryData.FromObjectAsJson(propertySchemaBuilder.Build(), option)
        };

        return functionDefinition;
    }
}
