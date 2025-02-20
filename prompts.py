def multiple_choice(question, options):
    choices = [f"{chr(65 + i)}) {options[i]}" for i in range(len(options))]

    prompt = f"Questions: {question}\n\nOptions:\n"

    for choice in choices:
        prompt += f"\n{choice}"

    prompt += "\n\nAnswer: "

    return prompt

def get_prompt(benchmark, **kwargs):
    if benchmark == "mmlu":
        return multiple_choice(**kwargs)
    if benchmark == "mmlu_pro":
        return multiple_choice(**kwargs)
