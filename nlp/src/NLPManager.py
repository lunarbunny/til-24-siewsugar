from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import json
import os

MODEL_CHKPT_NAME = "./models/bert-large-uncased-whole-word-masking-finetuned-squad"
# if os.path.exists(MODEL_CHKPT_NAME):
#     print(f"Directory '{MODEL_CHKPT_NAME}' exists.")
# else:
#     print(f"Directory '{MODEL_CHKPT_NAME}' does not exist.")

class NLPManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[NLP] Device: {self.device}")
        self.model_name = MODEL_CHKPT_NAME
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def qa(self, context: str) -> Dict[str, str]:
        questions = {
            "heading": "What is the heading?",
            "target": "What is the target?",
            "tool": "What is the tool to deploy?"
        }

        inputs = self.tokenizer([q for q in questions.values()], [context] * len(questions), add_special_tokens=True, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        extracted_info = {}
        for idx, key in enumerate(questions.keys()):
            answer_start = torch.argmax(start_logits[idx])
            answer_end = torch.argmax(end_logits[idx]) + 1
            input_ids = inputs["input_ids"][idx].tolist()
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])).strip()
            extracted_info[key] = answer
            if key == "tool":
                answer = self.normalize_tool(answer)
            extracted_info[key] = answer

        if extracted_info["heading"]:
            extracted_info["heading"] = self.words_to_nums(extracted_info["heading"])

        return extracted_info

    def normalize_tool(self, tool: str) -> str:
        return tool.replace(" - ", "-")

    def words_to_nums(self, heading: str) -> str:
        words = heading.split()
        heading_numbers = [str(word_to_number(word)) for word in words]
        return "".join(heading_numbers)

def word_to_number(word: str) -> int:
    word_to_num_map = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "niner": 9
    }
    if word in word_to_num_map:
        return word_to_num_map[word]
    else:
        print(f"Word '{word}' is not a recognized number")

## Check code
nlp_manager = NLPManager()
transcription = "Control Tower here. Deploy EMP, heading three six zero, target is grey and yellow cargo aircraft. Engage and neutralize. Over."
result = nlp_manager.qa(transcription)
print(result)