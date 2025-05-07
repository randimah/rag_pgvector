import datasets
from deepeval.metrics import (AnswerRelevancyMetric,
                              FaithfulnessMetric,
                              ContextualPrecisionMetric,
                              ContextualRecallMetric,
                              ContextualRelevancyMetric)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

# define the RAG metrics for evaluation with the threshold for pass/fail
answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
faithfulness_metric = FaithfulnessMetric(threshold=0.7)
contextual_precision_metric = ContextualPrecisionMetric(threshold=0.7)
contextual_recall_metric = ContextualRecallMetric(threshold=0.7)
contextual_relevancy_metric = ContextualRelevancyMetric(threshold=0.7)

def evaluate_generated_response(query,
                                actual_output,
                                expected_output,
                                context):
    test_case = LLMTestCase(
        input = query,
        actual_output = actual_output,
        expected_output= expected_output,
        retrieval_context = context
    )

    # evaluate the generated output against the test case
    # alternatively answer_relevancy_metric.measure(test_case) can be used
    # to calculate individual metrics
    evaluate(
        test_cases=[test_case],
        metrics=[
            answer_relevancy_metric,
            faithfulness_metric,
            contextual_precision_metric,
            contextual_recall_metric,
            contextual_relevancy_metric
        ])

def load_eval_dataset():
    ds = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
    return ds['question'], ds['answer']