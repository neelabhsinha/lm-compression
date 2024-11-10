from src.compression.context_compress import ContextCompressor


class PromptMap:
    def __init__(self, tokenizer, compress_context, max_context_size):
        self.prompts = {
            "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. "
                           "Answer the question as concisely as you can, using a single phrase if possible. "
                           "Do not provide any explanation.\nStory: {context}\nNow, answer the question based on "
                           "the story as concisely as you can, using a single phrase if possible. Do not provide any "
                           "explanation.\nQuestion: {input}\nAnswer:",
            "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, "
                      "using a single phrase or sentence if possible. If the question cannot be answered based on the "
                      "information in the article, write “unanswerable”. If the question is a yes/no question, answer "
                      "“yes”, “no”, or “unanswerable”. Do not provide any explanation.\nArticle: {context}\nAnswer the "
                      "question based on the above article as concisely as you can, using a single phrase or sentence "
                      "if possible. If the question cannot be answered based on the information in the article, write "
                      "“unanswerable”. If the question is a yes/no question, answer “yes”, “no”, or “unanswerable”. "
                      "Do not provide any explanation.\nQuestion: {input}\nAnswer:",
            "multifieldqa_en": "Read the following text and answer briefly.\n{context}\nNow, answer the following "
                               "question based on the above text, only give me the answer and do not output any other"
                               " words.\nQuestion: {input}\nAnswer:",
            "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output "
                        "any other words.\nThe following are given passages.\n{context}\nAnswer the question based on "
                        "the given passages. Only give me the answer and do not output any other words."
                        "\nQuestion: {input}\nAnswer:",
            "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do "
                        "not output any other words.\nThe following are given passages.\n{context}"
                        "\nAnswer the question based on the given passages. Only give me the answer and do not "
                        "output any other words.\nQuestion: {input}\nAnswer:",
            "musique": "Answer the question based on the given passages. Only give me the answer and do not output "
                       "any other words.\nThe following are given passages.\n{context}\nAnswer the question based on "
                       "the given passages. Only give me the answer and do not output any other words."
                       "\nQuestion: {input}\nAnswer:",
            "gov_report": "You are given a report by a government agency. Write a one-page summary of the report."
                          "\nReport: {context}\nNow, write a one-page summary of the report.\nSummary:",
            "qmsum": "You are given a meeting transcript and a query containing a question or instruction. "
                     "Answer the query in one or more sentences.\nTranscript: {context}\nNow, answer the query based "
                     "on the above meeting transcript in one or more sentences.\nQuery: {input}\nAnswer:",
            "multi_news": "You are given several news passages. Write a one-page summary of all news.\nNews: {context}"
                          "\nNow, write a one-page summary of all the news.\nSummary:",
            "trec": "Please determine the type of the question below. Here are some examples of questions."
                    "\n{context}\n{input}",
            "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output "
                        "any other words. The following are some examples.\n{context}\n{input}",
            "samsum": "Summarize the dialogue into a few short sentences. The following are some examples."
                      "\n{context}\n{input}",
            "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. "
                             "Please carefully read these paragraphs and determine how many unique paragraphs there "
                             "are after removing duplicates. In other words, how many non-repeating paragraphs are "
                             "there in total?\n{context}\nPlease enter the final count of unique paragraphs after "
                             "removing duplicates. The output format should only contain the number, such as 1, 2, 3, "
                             "and so on.\nThe final answer is:",
            "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine "
                                    "which paragraph the abstract is from.\n{context}\nThe following is an abstract."
                                    "\n{input}\nPlease enter the number of the paragraph that the abstract is from. "
                                    "The answer format must be like “Paragraph 1”, “Paragraph 2”, etc.\nThe answer is:",
            "lcc": "Please complete the code given below.\n{context}Next line of code:",
            "repobench-p": "Please complete the code given below.\n{context}{input}Next line of code:"
        }
        self.prompt_compressor = ContextCompressor(tokenizer, max_context_size) if compress_context else None

    def get_prompt_function(self, dataset_name):
        template = self.prompts.get(dataset_name)
        if template is None:
            raise ValueError(f"No prompt found for dataset: {dataset_name}")

        def prompt_fn(instance):
            context = instance.get("context", "")
            dataset = instance.get("dataset", "")
            compression_direction = "left" if dataset in ["lcc", "repobench-p"] else "right"
            if self.prompt_compressor is not None:
                context = self.prompt_compressor.compress(template, context, instance.get("input", ""), compression_direction)
            input_text = instance.get("input", "")
            prompt_text = template.format(context=context, input=input_text)
            instance["input_text"] = prompt_text
            instance["target"] = instance.get("answers", "")
            del instance["context"]
            del instance["input"]
            del instance["answers"]
            return instance

        return prompt_fn
