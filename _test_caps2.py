from browser_use.llm.anthropic.chat import ChatAnthropic
from browser_use.llm.google.chat import ChatGoogle
from browser_use.llm.deepseek.chat import ChatDeepSeek
from browser_use.llm.groq.chat import ChatGroq
from browser_use.llm.cerebras.chat import ChatCerebras
from browser_use.llm.oci_raw.chat import ChatOCIRaw
from browser_use.llm.capabilities import StructuredOutputMethod, extract_json_candidates, parse_structured_output_from_text
from pydantic import BaseModel

class TestModel(BaseModel):
    name: str
    value: int

# Test utility functions
text = "Some text ```json {\"name\": \"test\", \"value\": 42} ``` more text"
candidates = extract_json_candidates(text)
print(f"Extracted {len(candidates)} JSON candidates")
parsed = parse_structured_output_from_text(text, TestModel)
assert parsed is not None
assert parsed.name == "test"
assert parsed.value == 42
print("Utility function test passed!")

for name, cls, kwargs in [
    ("Anthropic", ChatAnthropic, {"model": "claude-sonnet-4-0", "api_key": "test"}),
    ("Google", ChatGoogle, {"model": "gemini-2.5-flash", "api_key": "test"}),
    ("DeepSeek", ChatDeepSeek, {"model": "deepseek-chat", "api_key": "test"}),
    ("Groq-tool", ChatGroq, {"model": "moonshotai/kimi-k2-instruct", "api_key": "test"}),
    ("Groq-json", ChatGroq, {"model": "meta-llama/llama-4-maverick-17b-128e-instruct", "api_key": "test"}),
    ("Cerebras", ChatCerebras, {"model": "llama3.1-8b", "api_key": "test"}),
]:
    llm = cls(**kwargs)
    chain = llm.capabilities.get_structured_output_strategy_chain()
    print(f"{name}: {[m.value for m in chain]}")

print("All tests passed!")
