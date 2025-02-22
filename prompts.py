import textwrap

def multiple_choice(question, options):
    choices = [f"{chr(65 + i)}) {options[i]}" for i in range(len(options))]

    prompt = f"Questions: {question}\n\nOptions:\n"

    for choice in choices:
        prompt += f"\n{choice}"

    prompt += "\n\nAnswer: "

    return prompt


def tabfact(question, table_text, table_caption):
    prompt = textwrap.dedent(f"""\
        Question: {question}

        Table:
        """)
    prompt += table_text
    prompt += textwrap.dedent(f"""\
                              
        Caption: {table_caption}

        Is the question entailed or refuted by the table?

        Options:

        A) Refuted
        B) Entailed

        Answer: 
    """)
    return prompt
    

def get_prompt(benchmark, **kwargs):
    if benchmark == "mmlu":
        return multiple_choice(**kwargs)
    if benchmark == "mmlu_pro":
        return multiple_choice(**kwargs)
    if benchmark == "tabfact":
        return tabfact(**kwargs)
