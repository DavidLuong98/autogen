"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8181],{971:(n,e,t)=>{t.r(e),t.d(e,{assets:()=>r,contentTitle:()=>i,default:()=>h,frontMatter:()=>a,metadata:()=>c,toc:()=>l});var o=t(5893),s=t(1151);const a={custom_edit_url:"https://github.com/microsoft/autogen/edit/main/notebook/agentchat_function_call_async.ipynb",description:"Learn how to implement both synchronous and asynchronous function calls using AssistantAgent and UserProxyAgent in AutoGen, with examples of their application in individual and group chat settings for task execution with language models.",source_notebook:"/notebook/agentchat_function_call_async.ipynb",tags:["code generation","function call","async"],title:"Task Solving with Provided Tools as Functions (Asynchronous Function Calls)"},i="Task Solving with Provided Tools as Functions (Asynchronous Function Calls)",c={id:"notebooks/agentchat_function_call_async",title:"Task Solving with Provided Tools as Functions (Asynchronous Function Calls)",description:"Learn how to implement both synchronous and asynchronous function calls using AssistantAgent and UserProxyAgent in AutoGen, with examples of their application in individual and group chat settings for task execution with language models.",source:"@site/docs/notebooks/agentchat_function_call_async.mdx",sourceDirName:"notebooks",slug:"/notebooks/agentchat_function_call_async",permalink:"/autogen/docs/notebooks/agentchat_function_call_async",draft:!1,unlisted:!1,editUrl:"https://github.com/microsoft/autogen/edit/main/notebook/agentchat_function_call_async.ipynb",tags:[{label:"code generation",permalink:"/autogen/docs/tags/code-generation"},{label:"function call",permalink:"/autogen/docs/tags/function-call"},{label:"async",permalink:"/autogen/docs/tags/async"}],version:"current",frontMatter:{custom_edit_url:"https://github.com/microsoft/autogen/edit/main/notebook/agentchat_function_call_async.ipynb",description:"Learn how to implement both synchronous and asynchronous function calls using AssistantAgent and UserProxyAgent in AutoGen, with examples of their application in individual and group chat settings for task execution with language models.",source_notebook:"/notebook/agentchat_function_call_async.ipynb",tags:["code generation","function call","async"],title:"Task Solving with Provided Tools as Functions (Asynchronous Function Calls)"},sidebar:"notebooksSidebar",previous:{title:"Task Solving with Code Generation, Execution and Debugging",permalink:"/autogen/docs/notebooks/agentchat_auto_feedback_from_code_execution"},next:{title:"Group Chat",permalink:"/autogen/docs/notebooks/agentchat_groupchat"}},r={},l=[{value:"Making Async and Sync Function Calls",id:"making-async-and-sync-function-calls",level:2}];function d(n){const e={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",img:"img",p:"p",pre:"pre",...(0,s.a)(),...n.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(e.h1,{id:"task-solving-with-provided-tools-as-functions-asynchronous-function-calls",children:"Task Solving with Provided Tools as Functions (Asynchronous Function Calls)"}),"\n",(0,o.jsxs)(e.p,{children:[(0,o.jsx)(e.a,{href:"https://colab.research.google.com/github/microsoft/autogen/blob/main/notebook/agentchat_function_call_async.ipynb",children:(0,o.jsx)(e.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})}),"\n",(0,o.jsx)(e.a,{href:"https://github.com/microsoft/autogen/blob/main/notebook/agentchat_function_call_async.ipynb",children:(0,o.jsx)(e.img,{src:"https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github",alt:"Open on GitHub"})})]}),"\n",(0,o.jsxs)(e.p,{children:["AutoGen offers conversable agents powered by LLM, tool, or human, which\ncan be used to perform tasks collectively via automated chat. This\nframework allows tool use and human participation through multi-agent\nconversation. Please find documentation about this feature\n",(0,o.jsx)(e.a,{href:"https://microsoft.github.io/autogen/docs/Use-Cases/agent_chat",children:"here"}),"."]}),"\n",(0,o.jsxs)(e.p,{children:["In this notebook, we demonstrate how to use ",(0,o.jsx)(e.code,{children:"AssistantAgent"})," and\n",(0,o.jsx)(e.code,{children:"UserProxyAgent"})," to make function calls with the new feature of OpenAI\nmodels (in model version 0613). A specified prompt and function configs\nmust be passed to ",(0,o.jsx)(e.code,{children:"AssistantAgent"})," to initialize the agent. The\ncorresponding functions must be passed to ",(0,o.jsx)(e.code,{children:"UserProxyAgent"}),", which will\nexecute any function calls made by ",(0,o.jsx)(e.code,{children:"AssistantAgent"}),". Besides this\nrequirement of matching descriptions with functions, we recommend\nchecking the system message in the ",(0,o.jsx)(e.code,{children:"AssistantAgent"})," to ensure the\ninstructions align with the function call descriptions."]}),"\n",(0,o.jsxs)(e.admonition,{title:"Requirements",type:"info",children:[(0,o.jsxs)(e.p,{children:["Install ",(0,o.jsx)(e.code,{children:"pyautogen"}),":"]}),(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-bash",children:"pip install pyautogen\n"})}),(0,o.jsxs)(e.p,{children:["For more information, please refer to the ",(0,o.jsx)(e.a,{href:"/docs/installation/",children:"installation guide"}),"."]})]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'import time\n\nfrom typing_extensions import Annotated\n\nimport autogen\nfrom autogen.cache import Cache\n\nconfig_list = autogen.config_list_from_json(env_or_file="OAI_CONFIG_LIST")\n'})}),"\n",(0,o.jsx)(e.admonition,{type:"tip",children:(0,o.jsxs)(e.p,{children:["Learn more about configuring LLMs for agents ",(0,o.jsx)(e.a,{href:"/docs/llm_configuration",children:"here"}),"."]})}),"\n",(0,o.jsx)(e.h2,{id:"making-async-and-sync-function-calls",children:"Making Async and Sync Function Calls"}),"\n",(0,o.jsxs)(e.p,{children:["In this example, we demonstrate function call execution with\n",(0,o.jsx)(e.code,{children:"AssistantAgent"})," and ",(0,o.jsx)(e.code,{children:"UserProxyAgent"}),". With the default system prompt of\n",(0,o.jsx)(e.code,{children:"AssistantAgent"}),", we allow the LLM assistant to perform tasks with code,\nand the ",(0,o.jsx)(e.code,{children:"UserProxyAgent"})," would extract code blocks from the LLM response\nand execute them. With the new \u201cfunction_call\u201d feature, we define\nfunctions and specify the description of the function in the OpenAI\nconfig for the ",(0,o.jsx)(e.code,{children:"AssistantAgent"}),". Then we register the functions in\n",(0,o.jsx)(e.code,{children:"UserProxyAgent"}),"."]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'llm_config = {\n    "config_list": config_list,\n}\n\ncoder = autogen.AssistantAgent(\n    name="chatbot",\n    system_message="For coding tasks, only use the functions you have been provided with. You have a stopwatch and a timer, these tools can and should be used in parallel. Reply TERMINATE when the task is done.",\n    llm_config=llm_config,\n)\n\n# create a UserProxyAgent instance named "user_proxy"\nuser_proxy = autogen.UserProxyAgent(\n    name="user_proxy",\n    system_message="A proxy for the user for executing code.",\n    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),\n    human_input_mode="NEVER",\n    max_consecutive_auto_reply=10,\n    code_execution_config={"work_dir": "coding"},\n)\n\n# define functions according to the function description\n\n# An example async function registered using register_for_llm and register_for_execution decorators\n\n\n@user_proxy.register_for_execution()\n@coder.register_for_llm(description="create a timer for N seconds")\nasync def timer(num_seconds: Annotated[str, "Number of seconds in the timer."]) -> str:\n    for i in range(int(num_seconds)):\n        time.sleep(1)\n        # should print to stdout\n    return "Timer is done!"\n\n\n# An example sync function registered using register_function\ndef stopwatch(num_seconds: Annotated[str, "Number of seconds in the stopwatch."]) -> str:\n    for i in range(int(num_seconds)):\n        time.sleep(1)\n    return "Stopwatch is done!"\n\n\nautogen.agentchat.register_function(\n    stopwatch,\n    caller=coder,\n    executor=user_proxy,\n    description="create a stopwatch for N seconds",\n)\n'})}),"\n",(0,o.jsxs)(e.p,{children:["Start the conversation. ",(0,o.jsx)(e.code,{children:"await"})," is used to pause and resume code\nexecution for async IO operations. Without ",(0,o.jsx)(e.code,{children:"await"}),", an async function\nreturns a coroutine object but doesn\u2019t execute the function. With\n",(0,o.jsx)(e.code,{children:"await"}),", the async function is executed and the current function is\npaused until the awaited function returns a result."]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'with Cache.disk() as cache:\n    await user_proxy.a_initiate_chat(  # noqa: F704\n        coder,\n        message="Create a timer for 5 seconds and then a stopwatch for 5 seconds.",\n        cache=cache,\n    )\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:'user_proxy (to chatbot):\n\nCreate a timer for 5 seconds and then a stopwatch for 5 seconds.\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\n***** Suggested tool Call (call_h6324df0CdGPDNjPO8GrnAQJ): timer *****\nArguments: \n{"num_seconds":"5"}\n**********************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> EXECUTING ASYNC FUNCTION timer...\nuser_proxy (to chatbot):\n\nuser_proxy (to chatbot):\n\n***** Response from calling tool "call_h6324df0CdGPDNjPO8GrnAQJ" *****\nTimer is done!\n**********************************************************************\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\n***** Suggested tool Call (call_7SzbQxI8Nsl6dPQtScoSGPAu): stopwatch *****\nArguments: \n{"num_seconds":"5"}\n**************************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> EXECUTING ASYNC FUNCTION stopwatch...\nuser_proxy (to chatbot):\n\nuser_proxy (to chatbot):\n\n***** Response from calling tool "call_7SzbQxI8Nsl6dPQtScoSGPAu" *****\nStopwatch is done!\n**********************************************************************\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\nTERMINATE\n\n--------------------------------------------------------------------------------\n'})}),"\n",(0,o.jsx)(e.h1,{id:"async-function-call-with-group-chat",children:"Async Function Call with Group Chat"}),"\n",(0,o.jsx)(e.p,{children:"Sync and async can be used in topologies beyond two agents. Below, we\nshow this feature for a group chat."}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'markdownagent = autogen.AssistantAgent(\n    name="Markdown_agent",\n    system_message="Respond in markdown only",\n    llm_config=llm_config,\n)\n\n# Add a function for robust group chat termination\n\n\n@user_proxy.register_for_execution()\n@markdownagent.register_for_llm()\n@coder.register_for_llm(description="terminate the group chat")\ndef terminate_group_chat(message: Annotated[str, "Message to be sent to the group chat."]) -> str:\n    return f"[GROUPCHAT_TERMINATE] {message}"\n\n\ngroupchat = autogen.GroupChat(agents=[user_proxy, coder, markdownagent], messages=[], max_round=12)\n\nllm_config_manager = llm_config.copy()\nllm_config_manager.pop("functions", None)\n\nmanager = autogen.GroupChatManager(\n    groupchat=groupchat,\n    llm_config=llm_config_manager,\n    is_termination_msg=lambda x: "GROUPCHAT_TERMINATE" in x.get("content", ""),\n)\n'})}),"\n",(0,o.jsx)(e.p,{children:"Finally, we initialize the chat that would use the functions defined\nabove:"}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'message = """\n1) Create a timer and a stopwatch for 5 seconds each in parallel.\n2) Pretty print the result as md.\n3) when 1 and 2 are done, terminate the group chat\n"""\n\nwith Cache.disk() as cache:\n    await user_proxy.a_initiate_chat(  # noqa: F704\n        manager,\n        message=message,\n        cache=cache,\n    )\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:'user_proxy (to chat_manager):\n\n\n1) Create a timer and a stopwatch for 5 seconds each in parallel.\n2) Pretty print the result as md.\n3) when 1 and 2 are done, terminate the group chat\n\n\n--------------------------------------------------------------------------------\nchatbot (to chat_manager):\n\n***** Suggested tool Call (call_qlS3QkcY1NkfgpKtCoR6oGo7): timer *****\nArguments: \n{"num_seconds": "5"}\n**********************************************************************\n***** Suggested tool Call (call_TEHlvMgCp0S3RzBbVsVPXWeL): stopwatch *****\nArguments: \n{"num_seconds": "5"}\n**************************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> EXECUTING ASYNC FUNCTION timer...\n\n>>>>>>>> EXECUTING ASYNC FUNCTION stopwatch...\nuser_proxy (to chat_manager):\n\nuser_proxy (to chat_manager):\n\n***** Response from calling tool "call_qlS3QkcY1NkfgpKtCoR6oGo7" *****\nTimer is done!\n**********************************************************************\n\n--------------------------------------------------------------------------------\nuser_proxy (to chat_manager):\n\n***** Response from calling tool "call_TEHlvMgCp0S3RzBbVsVPXWeL" *****\nStopwatch is done!\n**********************************************************************\n\n--------------------------------------------------------------------------------\nMarkdown_agent (to chat_manager):\n\n***** Suggested tool Call (call_JuQwvj4FigfvGyBeTMglY2ee): terminate_group_chat *****\nArguments: \n{"message":"Both timer and stopwatch have completed their countdowns. The group chat is now being terminated."}\n*************************************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> EXECUTING ASYNC FUNCTION terminate_group_chat...\nuser_proxy (to chat_manager):\n\nuser_proxy (to chat_manager):\n\n***** Response from calling tool "call_JuQwvj4FigfvGyBeTMglY2ee" *****\n[GROUPCHAT_TERMINATE] Both timer and stopwatch have completed their countdowns. The group chat is now being terminated.\n**********************************************************************\n\n--------------------------------------------------------------------------------\n'})})]})}function h(n={}){const{wrapper:e}={...(0,s.a)(),...n.components};return e?(0,o.jsx)(e,{...n,children:(0,o.jsx)(d,{...n})}):d(n)}},1151:(n,e,t)=>{t.d(e,{Z:()=>c,a:()=>i});var o=t(7294);const s={},a=o.createContext(s);function i(n){const e=o.useContext(a);return o.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function c(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(s):n.components||s:i(n.components),o.createElement(a.Provider,{value:e},n.children)}}}]);