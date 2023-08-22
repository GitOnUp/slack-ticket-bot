import logging
from typing import List, Union, Any, Optional, Callable

import slack_sdk
from pydantic import Field
from steamship import Block, Task, DocTag, Steamship
from steamship.agents.functional import FunctionsBasedAgent
from steamship.agents.llms.openai import ChatOpenAI
from steamship.agents.mixins.transports.slack import SlackTransport, SlackTransportConfig, SlackThreadingBehavior, \
    SlackContextBehavior
from steamship.agents.mixins.transports.steamship_widget import SteamshipWidgetTransport
from steamship.agents.schema import Tool, AgentContext, Metadata
from steamship.agents.service.agent_service import AgentService
from steamship.agents.utils import get_llm
from steamship.data import TagValueKey
from steamship.data.block import get_tag_value_key
from steamship.data.user import User
from steamship.utils.repl import AgentREPL


class SlackTransportWithHistory(SlackTransport):
    def _manifest(self) -> dict:
        # Dirty hack.  Design for how to do this in an otherwise componentized way is tricky to balance exposing too
        # much of the peanut butter of slack into steamship's chocolate though.
        manifest = super()._manifest()
        manifest["oauth_config"]["scopes"]["bot"].append("channels:history")
        return manifest

    def _send(self, blocks: List[Block], metadata: Metadata):
        logging.info(f"Sending: {repr(blocks[0])}")
        super()._send(blocks, metadata)


def block_thread_ts(block: Block) -> Optional[str]:
    return get_tag_value_key(block.tags, TagValueKey.STRING_VALUE, kind=DocTag.CHAT, name="slack-threadid")


class SlackTicketSummarizerTool(Tool):
    name: str = "SlackTicketSummarizer"
    human_description: str = "Summarizes Slack threads to be appropriate as a ticket in an issue tracking system"
    agent_description = """ Used to summarize Slack threads into a format suitable for creation as a ticket
    Input: A request to summarize a thread.  Input is assumed to be from Slack.
    Output: A summarized version of the slack conversation broken into Summary / Title.  The agent can then be asked to
    modify it in basic ways.
    """

    PROMPT_PREFIX = """Summarize this chat log into a title and description for a ticket in an issue tracking system.
    Ignore information that looks unrelated.  Do not specify what kind of issue it is in the title.\n\n"""

    token_provider: Callable = Field(description="For retrieving slack tokens.")

    def _client(self):
        return slack_sdk.WebClient(token=self.token_provider())

    def _get_thread_history(self, channel: str, thread_ts: str) -> str:
        response = self._client().conversations_replies(channel=channel, ts=thread_ts)
        response.validate()
        messages = response.get("messages", [])
        thread_messages = []
        for message in messages:
            if message["type"] != "message" or not message.get('text'):
                continue
            # Look into attaching images from these, possibly.
            thread_message = f"{message['user']}: {message['text']}"
            thread_messages.append(thread_message)
        return '\n'.join(thread_messages)

    def run(self, tool_input: List[Block], context: AgentContext) -> Union[List[Block], Task[Any]]:
        logging.info(f"input to run: {repr(tool_input)}")
        logging.info(f"context metadata: {repr(context.metadata)}")
        channel = context.metadata.get("slack-channel")
        thread_ts = context.metadata.get("slack-threadts")
        if not thread_ts:
            # This is awkward, I just want to have it respond with some predefined text without hitting the LLM?
            return [Block(text="Tell the user that you need to be invoked via pinging in a slack thread")]
        thread_history = self._get_thread_history(channel, thread_ts)
        return get_llm(context).complete(self.PROMPT_PREFIX + thread_history)


class MyAssistant(AgentService):
    USED_MIXIN_CLASSES = [SteamshipWidgetTransport, SlackTransportWithHistory]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Dirty Hack.  Wanted to focus on implementation before refactoring transport into greater concept of Slack
        # capabilities.  May want a general slack extras world with an easier way of getting this configuration too.
        class TokenProvider(Callable):
            slack_transport: Optional[SlackTransport] = None

            def __call__(self, *args, **kwargs) -> Optional[str]:
                if self.slack_transport:
                    return self.slack_transport.get_slack_access_token()
                return None

        token_provider = TokenProvider()
        ticket_summarizer = SlackTicketSummarizerTool(token_provider=token_provider)
        self.agent = FunctionsBasedAgent(llm=ChatOpenAI(self.client), tools=[ticket_summarizer])
        # This Mixin provides HTTP endpoints that connects this agent to a web client
        self.add_mixin(
            SteamshipWidgetTransport(client=self.client, agent_service=self),
        )
        config = SlackTransportConfig(
            threading_behavior=SlackThreadingBehavior.ALWAYS_THREADED,
            context_behavior=SlackContextBehavior.ENTIRE_CHANNEL
        )
        slack_transport = SlackTransportWithHistory(client=self.client, config=config, agent_service=self)
        self.add_mixin(slack_transport)
        token_provider.slack_transport = slack_transport


if __name__ == "__main__":
    # Enables IDE debugging
    client = Steamship()
    user = User.current(client)
    AgentREPL(MyAssistant, agent_package_config={},).run()
