import argparse
import os
import json
import time
import torch
import openai
from tqdm import tqdm

from llm.chatgpt import init_chatgpt, ask_llm
from utils.enums import LLM
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.post_process import process_duplication, get_sqls
from utils.utils import get_template, get_answer

QUESTION_FILE = "questions.json"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--openai_group_id", type=str, default="org-ktBefi7n9aK7sZjwc2R9G1Wo")
    parser.add_argument("--model", type=str, choices=[LLM for LLM in LLM.LLMS],
                        default=LLM.GPT_35_TURBO)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--mini_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n", type=int, default=5, help="Size of self-consistent set")
    parser.add_argument("--db_dir", type=str, default="/home/koushurui/Documents/Data/text2sql/spider/database")
    args = parser.parse_args()

    # check args
    assert args.model in LLM.BATCH_FORWARD or \
           args.model not in LLM.BATCH_FORWARD and args.batch_size == 1, \
        f"{args.model} doesn't support batch_size > 1"

    questions_json = json.load(open(os.path.join(args.question, QUESTION_FILE), "r"))
    questions = [_["prompt"] for _ in questions_json["questions"]]
    db_ids = [_["db_id"] for _ in questions_json["questions"]]

    question_loader = DataLoader(questions, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if args.mini_index > 0:
        questions = [questions[i] for i in range(args.mini_index)]
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}_MINI.txt"
    else:
        out_file = f"{args.question}/RESULTS_MODEL-{args.model}.txt"


    if args.start_index == 0:
        mode = "w"
    else:
        mode = "a"

    if args.model in LLM.OPEN_AI_LLM:
        # init openai api
        init_chatgpt(args.openai_api_key, args.openai_group_id, args.model)

        token_cnt = 0
        with open(out_file, mode) as f:
            for i, batch in enumerate(tqdm(question_loader)):
                if i < args.start_index:
                    continue
                if i >= args.end_index:
                    break
                try:
                    res = ask_llm(args.model, batch, args.temperature, args.n)
                except openai.error.OpenAIError as e:
                    print(f"The {i}-th question has too much tokens! Return \"SELECT\" instead")
                    res = ""

                # parse result
                token_cnt += res["total_tokens"]
                if args.n == 1:
                    for sql in res["response"]:
                        # remove \n and extra spaces
                        sql = " ".join(sql.replace("\n", " ").split())
                        sql = process_duplication(sql)
                        # python version should >= 3.8
                        if sql.startswith("SELECT"):
                            f.write(sql + "\n")
                        elif sql.startswith(" "):
                            f.write("SELECT" + sql + "\n")
                        else:
                            f.write("SELECT " + sql + "\n")
                else:
                    results = []
                    cur_db_ids = db_ids[i * args.batch_size: i * args.batch_size + len(batch)]
                    for sqls, db_id in zip(res["response"], cur_db_ids):
                        processed_sqls = []
                        for sql in sqls:
                            sql = " ".join(sql.replace("\n", " ").split())
                            sql = process_duplication(sql)
                            if sql.startswith("SELECT"):
                                pass
                            elif sql.startswith(" "):
                                sql = "SELECT" + sql
                            else:
                                sql = "SELECT " + sql
                            processed_sqls.append(sql)
                        result = {
                            'db_id': db_id,
                            'p_sqls': processed_sqls
                        }
                        final_sqls = get_sqls([result], args.n, args.db_dir)

                        for sql in final_sqls:
                            f.write(sql + "\n")
    elif args.model in LLM.OPEN_SOURCE_LLM:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        model_path = LLM.OPEN_SOURCE_2_PATH[args.model]
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
        model.eval()

        # with open("./results/re_text.txt", "w") as f1, open("./results/original_text.txt", "w") as f2:
        #     chat = get_template([questions[3]])
            
        #     chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=True)
        #     chat = chat.cuda()
        #     inputs = tokenizer.batch_decode(chat, skip_special_tokens=True) 
        #     for input in inputs:
        #         f2.write(input + '\n')
        #     outputs = model.generate(chat, max_new_tokens=2048)
        #     outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        #     for i, output in enumerate(outputs):
        #         try:
        #             ans = get_answer(output)
        #             f1.write(ans+'\n')
        #         except Exception as e:
        #             print(e)
        #         finally:
        #             f1.write(output)

        with open(out_file, "w") as f1, open("./results/original_text.txt", "w") as f2, open("./results/wrong_text.txt", "w") as f3:
            processor = tqdm(question_loader)
            for i, batch in enumerate(processor):
                s_time = time.time()
                try:
                    chat = get_template(batch)
                except Exception:
                    f3.write(batch[0])
                chat = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt", padding=True, truncation=True)
                chat = chat.cuda()
                outputs = model.generate(chat, max_new_tokens=256)
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                e1_time = time.time()
                for i, output in enumerate(outputs):
                    try:
                        ans = get_answer(output)
                        f1.write(ans + '\n')
                    except Exception as e:
                        f1.write('SELECT \n')
                    finally:
                        f2.write(output + '\n')
                e2_time = time.time()
                processor.set_description("%.3fs generation, %.3fs match" % (e1_time - s_time, e2_time - e1_time))

