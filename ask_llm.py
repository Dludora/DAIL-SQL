import argparse
import os
import json
import re

from tqdm import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseOutputParser



class DeepseekParser(BaseOutputParser):
    def parse(self, text):
        match =  re.search(r"```sql(.*?)```", text, flags=re.DOTALL)
        if match:
            return re.sub(r"\s+", ' ', match.group(1), flags=re.DOTALL).strip()
        else:
            return "SELECT"
        
        
class args:
    model = "deepseek-r1:14b"
    data_root = "/home/koushurui/Documents/Code/text2sql/DAIL-SQL/dataset/process/MINIDEV"
    language = "sqlite"
    

if __name__ == "__main__":
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"
    
    output_parser = DeepseekParser()
    llm = ChatOpenAI(model=args.model, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        """
        You are a sqlite expert.
        You are required to generate sqlite language based on the following question on this format:
        ```sql
        sqlite language
        ```
        """), 
        ("human", "{input}")])
    
    chain = prompt | llm | output_parser
    
    questions_json = json.load(open(os.path.join(args.data_root, args.language+"_data.json"), "r"))
    questions = [_["prompt"] for _ in questions_json]
    db_ids = [_["db_id"] for _ in questions_json]

    out_file = f"{args.data_root}/RESULTS/model-{args.model}-language-{args.language}.txt"
    
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        for i, question in enumerate(tqdm(questions_json)):
            f.write(chain.invoke(question["prompt"])+"\n")