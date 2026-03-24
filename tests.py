import json
import math
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from agents_service import (
    app,
    chunk_ssssstore,
    session_store,
    chunk_text,
    cosine_similarity,
    cosine_similarity_search,
    calculator_tool,
)

MOCK_EMBEDDING = [0.1] * 1536

SAMPLE_DOC = {
    "doc_id": "cs_notes",
    "text": (
        "Recursion is a programming technique where a function calls itself. "
        "Base case stops the recursion. Recursive algorithms are used in tree traversal. "
        "Dynamic programming can replace recursion for efficiency. "
        "Stack overflow can occur with deep recursion. "
        "Tail recursion is an optimization supported by some compilers. "
        "Fibonacci sequence is a classic recursion example. "
    ) * 20,
}


def make_test_client():
    chunk_store.clear()
    session_store.clear()
    return TestClient(app)


def make_mock_message(content=None, tool_calls=None, finish_reason="stop"):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    response = MagicMock()
    response.choices = [choice]
    return response


def make_tool_call(name, arguments: dict, call_id="call_abc123"):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


@pytest.fixture
def client_with_mocks():
    with patch("agent_service.AsyncOpenAI") as mock_cls, \
         patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-fake"}):

        mock_instance = AsyncMock()
        mock_instance.embeddings.create.return_value = AsyncMock(
            data=[AsyncMock(embedding=MOCK_EMBEDDING)]
        )
        mock_cls.return_value = mock_instance

        import agent_service
        agent_service._openai_client = None

        yield make_test_client(), mock_instance


class TestCalculatorTool:
    def test_addition(self):
        assert calculator_tool("2 + 3") == "5"

    def test_subtraction(self):
        assert calculator_tool("10 - 4") == "6"

    def test_multiplication(self):
        assert calculator_tool("25 * 48") == "1200"

    def test_division(self):
        assert calculator_tool("10 / 4") == "2.5"

    def test_integer_result_no_decimal(self):
        assert calculator_tool("9 / 3") == "3"

    def test_exponentiation(self):
        assert calculator_tool("2 ** 10") == "1024"

    def test_complex_expression(self):
        assert calculator_tool("(10 + 5) * 2") == "30"

    def test_division_by_zero(self):
        assert "Error" in calculator_tool("1 / 0")

    def test_invalid_expression(self):
        assert "Error" in calculator_tool("import os")

    def test_unsafe_expression_blocked(self):
        assert "Error" in calculator_tool("__import__('os').system('ls')")


class TestChunker:
    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_returns_single_chunk(self):
        chunks = chunk_text("This is a short document.", chunk_size=300, overlap=50)
        assert len(chunks) == 1

    def test_chunk_count_is_correct(self):
        text = " ".join(["word"] * 1000)
        assert len(chunk_text(text, chunk_size=300, overlap=50)) == 4

    def test_overlap_is_present(self):
        words = [f"w{i}" for i in range(400)]
        chunks = chunk_text(" ".join(words), chunk_size=200, overlap=50)
        assert chunks[0].split()[-50:] == chunks[1].split()[:50]

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            chunk_text(" ".join(["word"] * 500), chunk_size=100, overlap=100)


class TestCosineSimilarity:
    def test_identical_vectors(self):
        vec = [1.0, 2.0, 3.0]
        assert cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_search_top_k_order(self):
        store = {
            "doc#0": {"text": "a", "embedding": [1.0, 0.0]},
            "doc#1": {"text": "b", "embedding": [0.9, 0.1]},
            "doc#2": {"text": "c", "embedding": [0.0, 1.0]},
        }
        results = cosine_similarity_search([1.0, 0.0], store, k=2)
        assert results[0][0] == "doc#0"
        assert results[1][0] == "doc#1"


class TestIngestEndpoint:
    def test_ingest_returns_doc_id(self, client_with_mocks):
        client, _ = client_with_mocks
        resp = client.post("/ingest", json=SAMPLE_DOC)
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == "cs_notes"

    def test_ingest_returns_positive_chunk_count(self, client_with_mocks):
        client, _ = client_with_mocks
        resp = client.post("/ingest", json=SAMPLE_DOC)
        assert resp.json()["chunks_added"] > 0

    def test_ingest_short_doc_one_chunk(self, client_with_mocks):
        client, _ = client_with_mocks
        resp = client.post("/ingest", json={"doc_id": "tiny", "text": "Short document."})
        assert resp.json()["chunks_added"] == 1


class TestSearchEndpoint:
    def test_search_without_docs_404(self, client_with_mocks):
        client, _ = client_with_mocks
        assert client.get("/search?query=recursion").status_code == 404

    def test_search_returns_results(self, client_with_mocks):
        client, _ = client_with_mocks
        client.post("/ingest", json=SAMPLE_DOC)
        resp = client.get("/search?query=recursion&k=3")
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 3

    def test_search_k_limits_results(self, client_with_mocks):
        client, _ = client_with_mocks
        client.post("/ingest", json=SAMPLE_DOC)
        for k in [1, 2, 3]:
            resp = client.get(f"/search?query=test&k={k}")
            assert len(resp.json()["results"]) == k


class TestAgentEndpoint:
    def test_agent_simple_direct_answer(self, client_with_mocks):
        client, mock_instance = client_with_mocks
        mock_instance.chat.completions.create.return_value = make_mock_message(
            content="Hello! How can I help you?",
            finish_reason="stop",
        )
        resp = client.post("/agent", json={"session_id": "s1", "query": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == "Hello! How can I help you?"
        assert resp.json()["steps"] == []

    def test_agent_uses_calculator_for_math(self, client_with_mocks):
        client, mock_instance = client_with_mocks
        tool_call = make_tool_call("calculator", {"expression": "25 * 48"})
        first_response = make_mock_message(tool_calls=[tool_call], finish_reason="tool_calls")
        second_response = make_mock_message(content="25 * 48 = 1200", finish_reason="stop")
        mock_instance.chat.completions.create.side_effect = [first_response, second_response]

        resp = client.post("/agent", json={"session_id": "s2", "query": "What is 25 * 48?"})
        assert resp.status_code == 200
        steps = resp.json()["steps"]
        assert len(steps) == 1
        assert steps[0]["action"] == "calculator"
        assert steps[0]["input"] == "25 * 48"
        assert steps[0]["output"] == "1200"

    def test_agent_uses_knowledge_base_for_factual_query(self, client_with_mocks):
        client, mock_instance = client_with_mocks
        client.post("/ingest", json=SAMPLE_DOC)

        tool_call = make_tool_call("knowledge_base", {"query": "What is recursion?", "k": 3})
        first_response = make_mock_message(tool_calls=[tool_call], finish_reason="tool_calls")
        second_response = make_mock_message(content="Recursion is when a function calls itself.", finish_reason="stop")
        mock_instance.chat.completions.create.side_effect = [first_response, second_response]

        resp = client.post("/agent", json={"session_id": "s3", "query": "What is recursion?"})
        assert resp.status_code == 200
        steps = resp.json()["steps"]
        assert len(steps) == 1
        assert steps[0]["action"] == "knowledge_base"

    def test_agent_uses_multiple_tools_for_mixed_query(self, client_with_mocks):
        client, mock_instance = client_with_mocks
        client.post("/ingest", json=SAMPLE_DOC)

        calc_call = make_tool_call("calculator", {"expression": "25 * 48"}, call_id="c1")
        kb_call = make_tool_call("knowledge_base", {"query": "recursion"}, call_id="c2")

        first_response = make_mock_message(tool_calls=[calc_call], finish_reason="tool_calls")
        second_response = make_mock_message(tool_calls=[kb_call], finish_reason="tool_calls")
        third_response = make_mock_message(content="25*48=1200. Recursion is self-calling.", finish_reason="stop")
        mock_instance.chat.completions.create.side_effect = [first_response, second_response, third_response]

        resp = client.post("/agent", json={"session_id": "s4", "query": "What is 25*48 and explain recursion?"})
        assert resp.status_code == 200
        actions = [s["action"] for s in resp.json()["steps"]]
        assert "calculator" in actions
        assert "knowledge_base" in actions

    def test_agent_turn_count_increments(self, client_with_mocks):
        client, mock_instance = client_with_mocks
        mock_instance.chat.completions.create.return_value = make_mock_message(
            content="Answer.", finish_reason="stop"
        )
        session_id = "s-turns"
        for expected in range(1, 4):
            resp = client.post("/agent", json={"session_id": session_id, "query": "Hello"})
            assert resp.json()["turn_count"] == expected

    def test_agent_session_auto_created(self, client_with_mocks):
        client, mock_instance = client_with_mocks
        mock_instance.chat.completions.create.return_value = make_mock_message(
            content="Hi!", finish_reason="stop"
        )
        resp = client.post("/agent", json={"query": "Hello"})
        assert resp.status_code == 200
        assert resp.json()["turn_count"] == 1

    def test_agent_knowledge_base_empty_store(self, client_with_mocks):
        client, mock_instance = client_with_mocks
        tool_call = make_tool_call("knowledge_base", {"query": "recursion"})
        first_response = make_mock_message(tool_calls=[tool_call], finish_reason="tool_calls")
        second_response = make_mock_message(content="No info found.", finish_reason="stop")
        mock_instance.chat.completions.create.side_effect = [first_response, second_response]

        resp = client.post("/agent", json={"session_id": "s5", "query": "What is recursion?"})
        assert resp.status_code == 200
        assert "No documents" in resp.json()["steps"][0]["output"]

    def test_agent_step_fields_present(self, client_with_mocks):
        client, mock_instance = client_with_mocks
        tool_call = make_tool_call("calculator", {"expression": "1 + 1"})
        first_response = make_mock_message(tool_calls=[tool_call], finish_reason="tool_calls")
        second_response = make_mock_message(content="1+1=2", finish_reason="stop")
        mock_instance.chat.completions.create.side_effect = [first_response, second_response]

        resp = client.post("/agent", json={"session_id": "s6", "query": "1+1"})
        step = resp.json()["steps"][0]
        assert "action" in step
        assert "input" in step
        assert "output" in step
