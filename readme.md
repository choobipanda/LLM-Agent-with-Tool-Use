Reflection Questions
7. What is an LLM agent?
An LLM agent is a system where the language model does more than just generate a response. It can decide when to use tools, call those tools, observe the results, and then combine them into a final answer.

8. Why not always call tools?
Not every query needs a tool, and calling tools unnecessarily can make the system slower and more expensive. Simple questions can often be answered directly by the LLM without extra steps.

9. What are risks of tool misuse?
Tool misuse can lead to incorrect or unsafe results, especially if the wrong tool is selected or if inputs are not validated properly. For example, an unsafe calculator could allow harmful code execution instead of only arithmetic.

10. How does prompt design affect decisions?
Prompt design affects how the LLM decides whether to use a tool and which tool to select. A clear prompt with explicit rules makes the agent more likely to choose the correct action and avoid unnecessary tool calls.

11. What are limitations of this agent?
This agent is limited by the quality of its prompts, the reliability of its tools, and the accuracy of its reasoning. It may choose the wrong tool, misinterpret a query, or give weak answers if the retrieved knowledge base results are not relevant.
