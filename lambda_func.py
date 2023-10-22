from modal import Stub, Image, Secret, web_endpoint
from pydantic import BaseModel

stub = Stub()
CACHE_PATH = "/root/model_cache"

def download_model_weights() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", cache_dir=CACHE_PATH)

generate_qna_image = Image.debian_slim().pip_install("langchain", "pinecone_client", "sentence-transformers", "cohere", "huggingface_hub").run_function(download_model_weights)
evaluate_answer_image = Image.debian_slim().pip_install("langchain", "cohere")

class TopicRequest(BaseModel):
    topic: str

class EvaluationRequest(BaseModel):
    topic: str
    question: str
    examiner_answer: str
    user_answer: str


@stub.function(image=generate_qna_image, secret=Secret.from_name("ml-interview-warmup"))
@web_endpoint(method="GET")
def generate_qna(item: TopicRequest):
    import os
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chat_models import ChatCohere
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import PromptTemplate
    from langchain.pydantic_v1 import BaseModel, Field, validator

    # Define desired data structure for LLM output.
    class QnA(BaseModel):
        question: str = Field(description="question about a topic")
        answer: str = Field(description="answer for the question")

        # You can add custom validation logic easily with Pydantic.
        @validator('question', allow_reuse=True)
        def question_ends_with_question_mark(cls, field):
            if field[-1] != '?':
                raise ValueError("Badly formed question!")
            return field

    def get_retriever_from_vectorstore():
        PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
        PINECONE_ENV = os.environ["PINECONE_ENV"]

        # initialize pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,  # find at app.pinecone.io
            environment=PINECONE_ENV,  # next to api key in console
        )
        # get the vectorstore
        index_name = "ml-interview-warmup"
        embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", cache_folder=CACHE_PATH)

        index = pinecone.Index(index_name)
        vectorstore = Pinecone(index, embedding_function, "text")
        retriever = vectorstore.as_retriever()

        return retriever

    def get_prompt_template_and_parser():
        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=QnA)

        template = """Topic: {user_topic}
        Context: {context}

        Based on the context above, generate a question and answer pair about the topic.
        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["user_topic", "context"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        return prompt, parser


    retriever = get_retriever_from_vectorstore()
    prompt, parser = get_prompt_template_and_parser()
    model = ChatCohere()
    chain = prompt | model | parser

    user_topic = item.topic
    relevant_docs = retriever.invoke(user_topic)

    output = chain.invoke({"user_topic": user_topic, "context": relevant_docs})


    return {
            "topic": user_topic,
            "question": output.question,
            "examiner_answer": output.answer
    }


@stub.function(image=evaluate_answer_image, secret=Secret.from_name("ml-interview-warmup"))
@web_endpoint(method="GET")
def evaluate_answer(item: EvaluationRequest):
    from langchain.chat_models import ChatCohere
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import PromptTemplate
    from langchain.pydantic_v1 import BaseModel, Field

    # Define desired data structure for LLM output.
    class AnswerEvaluation(BaseModel):
        score: int = Field(description="score for user's answer")
        justification: str = Field(description="justification explaining the score")

    def get_prompt_template_and_parser():
        # Set up a parser + inject instructions into the prompt template.
        parser = PydanticOutputParser(pydantic_object=AnswerEvaluation)

        template = """Topic: {user_topic}
        Question: {question}
        Examiner's answer: {examiner_answer}
        User's answer: {user_answer}

        Please act as an examiner and objectively rate the user's answer on the following 1-5 scale:
        - 1: Completely incorrect, irrelevant, or blank answer
        - 2: Demonstrates very limited understanding; major gaps in knowledge and logic
        - 3: Partial knowledge and major errors or omissions in reasoning
        - 4: Accurate but incomplete, with minor errors or omissions; lacks depth
        - 5: Accurate, reasonably complete, and demonstrates solid understanding

        Based on the question, examiner's answer and user's answer above, objectively provide a score between 1-5 for the user and a short justification explaining the score:
        {format_instructions}
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["user_topic", "question", "examiner_answer", "user_answer"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        return prompt, parser


    prompt, parser = get_prompt_template_and_parser()
    model = ChatCohere()
    chain = prompt | model | parser


    user_topic = item.topic
    question = item.question
    examiner_answer = item.examiner_answer
    user_answer = item.user_answer

    output = chain.invoke({"user_topic": user_topic, "question": question, "examiner_answer": examiner_answer, "user_answer": user_answer})

    return {
            "score": output.score,
            "justification": output.justification,
    }