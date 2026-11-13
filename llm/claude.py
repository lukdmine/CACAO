"""ChatAnthropic for the Claude models released after Opus 4.6 / Sonnet 4.6 / Haiku 4.5."""

from operator import itemgetter

from anthropic import transform_schema
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough


class ChatClaude(ChatAnthropic):
    """Structured output through ``output_config.format`` instead of a forced tool call.

    langchain gets structured output by forcing a call to a tool built from the schema
    (``tool_choice`` "tool"). Opus 5.5, Sonnet 5.5 and Fable 5.1 reject that with a 400,
    and every model this class is used for accepts ``output_config.format``, so they all
    take the one path.
    """

    def with_structured_output(self, schema, *, include_raw=False):
        output_format = {"type": "json_schema", "schema": transform_schema(schema)}
        llm = self.bind(output_config={"format": output_format})
        parser = PydanticOutputParser(pydantic_object=schema)
        if not include_raw:
            return llm | parser

        # langchain's include_raw contract, which TrackedLLM asks for so the reply's
        # tokens can be counted: {"raw", "parsed", "parsing_error"}, with a parse
        # failure reported rather than raised.
        parse = RunnablePassthrough.assign(
            parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
        )
        unparsed = RunnablePassthrough.assign(parsed=lambda _: None)
        return RunnableMap(raw=llm) | parse.with_fallbacks(
            [unparsed], exception_key="parsing_error"
        )
