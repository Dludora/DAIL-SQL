# echo "data_preprocess"
# python data_preprocess.py

echo "generate question with EUCDISQUESTIONMASK"
python generate_question.py \
--data_type bird \
--split train \
--tokenizer gpt-3.5-turbo \
--max_seq_len 4096 \
--prompt_repr SQL \
--k_shot 0 \
--example_type QA \
--selector_type  EUCDISQUESTIONMASK

# echo "generate SQL by codellama-7b-instruct for EUCDISMASKPRESKLSIMTHR as the pre-generated SQL query"
# python ask_llm.py \
# --model codellama-7b-instruct \
# --question ./dataset/process/SPIDER-TEST_SQL_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096_test \
# --batch_size 1

# echo "generate question with EUCDISMASKPRESKLSIMTHR"
# python generate_question.py \
# --data_type spider \
# --split test \
# --tokenizer gpt-3.5-turbo \
# --max_seq_len 4096 \
# --selector_type EUCDISMASKPRESKLSIMTHR \
# --pre_test_result ./dataset/process/SPIDER-TEST_SQL_9-SHOT_EUCDISQUESTIONMASK_QA-EXAMPLE_CTX-200_ANS-4096_test/RESULTS_MODEL-codellama-7b-instruct.txt \
# --prompt_repr SQL \
# --k_shot 9 \
# --example_type QA

# echo "generate SQL by codellama-7b-instruct for EUCDISMASKPRESKLSIMTHR"
# python ask_llm.py \
# --model codellama-7b-instruct \
# --question ./dataset/process/SPIDER-TEST_SQL_9-SHOT_EUCDISMASKPRESKLSIMTHR_QA-EXAMPLE_CTX-200_ANS-4096_test/ \
# --batch_size 1