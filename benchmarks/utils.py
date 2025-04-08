import pandas as pd

def serialize_sentence(table: pd.DataFrame) -> str:
    sentences = []
    for i, row in table.iterrows():
        sentence = f"Row {i + 1}: "
        statements = []
        for col in table.columns:
            statements.append(f"{col} is {row[col]}")
        
        sentence += ", ".join(statements)
        sentence += "."
        sentences.append(sentence.strip())
    return "\n".join(sentences)