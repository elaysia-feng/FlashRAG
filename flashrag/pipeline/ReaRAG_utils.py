import re
from termcolor import colored

class AgentUtilsBase():
    def __init__(self):
        pass

    def truncate(self, tokenizer, tokenized_prompt, model_max_length):
        half = int(model_max_length/2)
        prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=False) + \
                tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=False)
        
        return prompt

    def preprocess_query(self, query):
        if "'" in query and '"' in query:
            query = query.replace("'", "\\'").replace('"', '\\"')
        return query

    def extract_code(self, text: str) -> str:
        triple_match = re.search(r'```[^\n]*\n(.+?)```', text, re.DOTALL)
        single_match = re.search(r'`([^`]*)`', text, re.DOTALL)
        if triple_match:
            return triple_match.group(1)
        elif single_match:
            return single_match.group(1)
        return text
    
    def postprocess_agent_response(self, response):
        """
        Implement the postprocess_agent_response function
        """
        # 提取推理步骤中的动作，按实现检查 function 和 parameters 字段。
        raise NotImplementedError("postprocess_agent_response function must be implemented in a subclass")

class AgentUtils(AgentUtilsBase):
    def __init__(self):
        super().__init__()

    def parse_reasoning_steps(self, text: str):
        """
        Parse a string containing Thought/Action/Observation steps (including multi-line)
        and return a list of dictionaries of the form:
        
        [
            {
                "1": {
                    "Thought": "...",
                    "Action": "...",  # Only content inside backticks (if present)
                    "Observation": "..."
                }
            },
            {
                "2": {
                    "Thought": "...",
                    "Action": "...",
                    "Observation": "..."
                }
            },
            ...
        ]
        """
        # 解析带编号的 Thought、Action 和 Observation，并按步骤顺序整理。
        # Regex pattern to match lines that start with "Thought X:", "Action X:", or "Observation X:".
        # 匹配以带编号的 Thought、Action 或 Observation 开头的行。
        pattern = re.compile(r'^(Thought|Action|Observation)\s+(\d+):', re.MULTILINE)

        # This dictionary will accumulate:
        # 用这个字典累积各步骤的内容。
        # data_dict[step_number] = {"Thought": ..., "Action": ..., "Observation": ...}
        data_dict = {}

        # We'll track the current label (Thought/Action/Observation) and step number
        # 跟踪当前字段标签与步骤编号。
        current_label = None
        current_step = None
        last_pos = 0

        # Find all pattern occurrences in the text
        # 找出文本中所有匹配位置。
        matches = list(pattern.finditer(text))

        for match in matches:
            # If we already have a label in progress, we can record its content
            # 已经存在正在处理的标签时，先记录其内容。
            if current_label is not None:
                # Slice the text from the last match's end to the start of this new match
                # 截取上次匹配结束到本次匹配开始之间的文本。
                content = text[last_pos:match.start()].strip()
                # Store that content in data_dict
                # 将截取内容保存到对应步骤与字段。
                data_dict[current_step][current_label] = content

            # Extract the new label and step
            # 提取新的字段标签和步骤编号。
            label = match.group(1)       # "Thought", "Action", or "Observation"
            # 字段可以是 Thought、Action 或 Observation。
            step = match.group(2)        # e.g. "1", "2", "3"

            # Ensure a dict for this step
            # 确保当前步骤已建立对应字典。
            if step not in data_dict:
                data_dict[step] = {"Thought": None, "Action": None, "Observation": None}

            # Update current label/step
            # 更新当前字段和步骤。
            current_label = label
            current_step = step
            # We'll slice from here next time
            # 下一次从当前匹配的末尾继续截取。
            last_pos = match.end()

        # Handle the final block after the last match
        # 处理最后一次匹配之后的文本块。
        if current_label is not None:
            content = text[last_pos:].strip()
            data_dict[current_step][current_label] = content

        # Post-process:
        # 后处理。
        #  - For each step, extract only the text inside triple backticks for "Action".
        # 对每个步骤，只提取 Action 中反引号包裹的代码内容。
        for step_number in data_dict:
            action_text = data_dict[step_number]["Action"]
            if action_text:
                # Extract content inside triple backticks
                # 提取三反引号内部的内容。
                data_dict[step_number]["Action"] = self.extract_code(action_text)

        # Convert our dictionary to the desired list-of-dicts structure
        # 将中间字典转换为目标列表套字典结构。
        structured_data = []
        for step_number in sorted(data_dict.keys(), key=lambda x: int(x)):
            structured_data.append({step_number: data_dict[step_number]})

        return structured_data
    
    def postprocess_agent_response(self, response):
        """
        Extract Thought and Action, then extract the dict from Action.

        Return:
        - thought: str
        - action: dict
        - is_valid: bool
        """
        # 提取推理步骤中的动作，按实现检查 function 和 parameters 字段。
        parsed_codes = self.parse_reasoning_steps(response)

        thoughts = []
        actions = []
        for steps in parsed_codes:
            for step_idx, step in steps.items():
                thought = f"Thought {step_idx}: {step['Thought']}"
                action = eval(self.extract_code(step['Action']))

                assert "function" in action, f"Action does not contain 'function' key: {action}"
                assert "parameters" in action, f"Action does not contain 'parameters' key: {action}"

                thoughts.append(thought)
                actions.append(action)

        return thoughts, actions
    
def print_code(codes):
    for idx, step in enumerate(codes):
        print(f"{colored(step['thought'], 'blue')}")
        print(colored(f"Action {idx+1}:\n```\n{step['action']}\n```", 'red'))
        print(colored(f"Observation {idx+1}: {step['observation']}", 'yellow'))
        # print(f"{'-'*60}")
