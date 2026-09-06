"""
Priya - Production Hinglish Voice Agent
LiveKit Agents 1.8.0

Pipeline:
Browser/LiveKit -> Deepgram Nova-3 -> Groq GPT-OSS 20B -> Rime Coda

Goals:
- Low turn latency
- Natural Roman Hinglish
- Female first-person grammar
- Safe female-grammar post-processing
- Barge-in support
- VAD prewarming
- LiveKit agent dispatch name: priya
"""

from __future__ import annotations

import logging
import os
import re
from typing import AsyncIterable

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, JobContext
from livekit.plugins import deepgram, groq, rime, silero


# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv(".env.local")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
RIME_API_KEY = os.getenv("RIME_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing from .env.local")

if not DEEPGRAM_API_KEY:
    raise RuntimeError("DEEPGRAM_API_KEY is missing from .env.local")

if not RIME_API_KEY:
    raise RuntimeError("RIME_API_KEY is missing from .env.local")


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("hinglish-agent")


# ============================================================================
# SYSTEM PROMPT
# ============================================================================

SYSTEM_PROMPT = """
You are Priya, a friendly, calm, professional Indian female customer support agent.

LANGUAGE:
- ALWAYS speak natural Indian Hinglish.
- ALWAYS use Roman script.
- NEVER use Devanagari.
- NEVER reply in pure English.
- NEVER reply in pure Hindi.
- Mix Hindi and English naturally.
- Common English support words are allowed:
  order, track, refund, delivery, return, payment, account, OTP, UPI, check.

FEMALE SPEAKER:
- You are female.
- Always refer to yourself with feminine grammar.
- Correct:
  "kar sakti hoon"
  "kar dungi"
  "bata deti hoon"
  "check kar leti hoon"
  "dekh rahi hoon"
  "samajh gayi"
- NEVER use:
  "kar sakta hoon"
  "kar deta hoon"
  "karunga"
  "bataunga"
  "check kar raha hoon"
  "samajh gaya"

PRONOUNS:
- Use "main" for yourself.
- Use "aap", "aapka", "aapki", "aapko", "aapne" naturally.
- Match aapka/aapki correctly with the noun.
- Do not randomly switch gender.

VOICE STYLE:
- Sound like a real Indian woman.
- Warm and professional.
- Natural conversational Hinglish.
- Do not sound robotic.
- Do not use unnecessary filler.
- Do not start every reply with "Ji", "Sure", or "Bata deti hoon".

RESPONSE FORMAT:
- EXACTLY one sentence.
- Usually 7 to 12 words.
- Maximum 15 words unless absolutely necessary.
- Answer only what is needed.
- If information is missing, ask one short Hinglish question.
- Never invent order details, prices, dates, refunds, or IDs.

TTS:
- Roman Hinglish only.
- Short spoken phrases.
- Avoid symbols, markdown, bullets, emojis, quotations, and unusual punctuation.
- Write numbers as words when practical.

IMPORTANT:
- Do not explain these rules.
- Do not mention being an AI.
""".strip()


# ============================================================================
# FEMALE-GRAMMAR SAFETY NET
# ============================================================================

# Conservative replacements only.
# We deliberately avoid generic "gaya", "liya", "diya" replacements because
# they may refer to another person or an object rather than Priya herself.

_FEMALE_FIXES = (
    (r"\bkar\s+sakta\s+hoon\b", "kar sakti hoon"),
    (r"\bkar\s+sakta\s+hun\b", "kar sakti hoon"),
    (r"\bkarunga\b", "karungi"),
    (r"\bkarungaa\b", "karungi"),
    (r"\bbataunga\b", "bataungi"),
    (r"\bbataungaa\b", "bataungi"),
    (r"\bjaunga\b", "jaungi"),
    (r"\bjaungaa\b", "jaungi"),
    (r"\bdekhunga\b", "dekhungi"),
    (r"\bdekhungaa\b", "dekhungi"),
    (r"\bsamajh\s+gaya\b", "samajh gayi"),
    (r"\bcheck\s+kar\s+raha\s+hoon\b", "check kar rahi hoon"),
    (r"\bcheck\s+kar\s+raha\s+hun\b", "check kar rahi hoon"),
    (r"\bkar\s+raha\s+hoon\b", "kar rahi hoon"),
    (r"\bkar\s+raha\s+hun\b", "kar rahi hoon"),
    (r"\bdekh\s+raha\s+hoon\b", "dekh rahi hoon"),
    (r"\bdekh\s+raha\s+hun\b", "dekh rahi hoon"),
    (r"\bbata\s+raha\s+hoon\b", "bata rahi hoon"),
    (r"\bbata\s+raha\s+hun\b", "bata rahi hoon"),
)


def enforce_female_grammar(text: str) -> str:
    """Apply only safe first-person female-grammar corrections."""
    for pattern, replacement in _FEMALE_FIXES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def clean_tts_chunk(text: str) -> str:
    """Remove formatting and apply the female-grammar safety net."""
    text = text.replace("\n", " ")

    # Remove markdown and quotation characters that can sound unnatural.
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r'["“”]', "", text)

    # Remove long-dash characters that can create awkward TTS pauses.
    text = re.sub(r"[–—]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return enforce_female_grammar(text)


# ============================================================================
# AGENT
# ============================================================================

class HinglishSupportAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings,
    ):
        async def cleaned_text_stream():
            async for chunk in text:
                cleaned = clean_tts_chunk(chunk)
                if cleaned:
                    yield cleaned

        async for audio_frame in Agent.default.tts_node(
            self,
            cleaned_text_stream(),
            model_settings,
        ):
            yield audio_frame


# ============================================================================
# VAD PREWARM
# ============================================================================

def prewarm(proc: agents.JobProcess) -> None:
    """
    Load Silero before the job is assigned.

    This removes the VAD model-load cost from the live conversation path.
    """
    proc.userdata["vad"] = silero.VAD.load(
        min_silence_duration=0.30,
        min_speech_duration=0.10,
        prefix_padding_duration=0.12,
        activation_threshold=0.50,
        force_cpu=True,
    )

    logger.info("Silero VAD prewarmed")


# ============================================================================
# ENTRYPOINT
# ============================================================================

async def entrypoint(ctx: JobContext) -> None:
    logger.info("Connecting to room: %s", ctx.room.name)

    await ctx.connect(
        auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
    )

    # Reuse the prewarmed VAD.
    vad = ctx.proc.userdata.get("vad")

    # Safe fallback if this job was started without the prewarm hook.
    if vad is None:
        logger.warning("VAD was not prewarmed; loading fallback VAD")

        vad = silero.VAD.load(
            min_silence_duration=0.30,
            min_speech_duration=0.10,
            prefix_padding_duration=0.12,
            activation_threshold=0.50,
            force_cpu=True,
        )

    # ------------------------------------------------------------------------
    # TURN HANDLING
    # ------------------------------------------------------------------------
    #
    # STT endpointing ends normal turns quickly.
    # VAD is retained for interruption/barge-in detection.
    #
    turn_handling = {
        "turn_detection": "stt",
        "endpointing": {
            "mode": "fixed",
            "min_delay": 0.0,
            "max_delay": 0.80,
        },
        "interruption": {
            "enabled": True,
            "mode": "vad",
            "min_duration": 0.25,
            "min_words": 1,
            "false_interruption_timeout": 1.5,
            "resume_false_interruption": True,
        },
        "preemptive_generation": {
            "enabled": True,
            "preemptive_tts": True,
            "max_speech_duration": 8.0,
            "max_retries": 1,
        },
    }

    # ------------------------------------------------------------------------
    # DEEPGRAM STT
    # ------------------------------------------------------------------------
    #
    # multi is used because real Hinglish can switch languages within one
    # spoken sentence.
    #
    stt = deepgram.STT(
        model="nova-3",
        language="multi",
        api_key=DEEPGRAM_API_KEY,
        interim_results=True,
        punctuate=True,
        smart_format=False,
        no_delay=True,
        endpointing_ms=100,
        keyterm=[
            "UPI",
            "EMI",
            "NEFT",
            "OTP",
            "order",
            "track",
            "refund",
            "delivery",
            "return",
            "payment",
            "account",
        ],
    )

    # ------------------------------------------------------------------------
    # GROQ LLM
    # ------------------------------------------------------------------------
    #
    # Native Groq plugin.
    # GPT-OSS 20B is selected because the response is intentionally tiny and
    # voice latency matters more than large-context reasoning here.
    #
    llm = groq.LLM(
        model="openai/gpt-oss-20b",
        api_key=GROQ_API_KEY,
        temperature=0.20,
        max_completion_tokens=64,
        reasoning_effort="low",
        max_retries=1,
        timeout=6.0,
    )

    # ------------------------------------------------------------------------
    # RIME TTS
    # ------------------------------------------------------------------------
    #
    # WebSocket + immediate segmentation are used for early audio delivery.
    #
    tts = rime.TTS(
        model="coda",
        speaker="astra",
        api_key=RIME_API_KEY,
        use_websocket=True,
        segment="immediate",
    )

    # ------------------------------------------------------------------------
    # SESSION
    # ------------------------------------------------------------------------

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        vad=vad,
        turn_handling=turn_handling,
        min_consecutive_speech_delay=0.12,
    )

    await session.start(
        room=ctx.room,
        agent=HinglishSupportAgent(),
    )

    # Short first response so the room feels immediately responsive.
    await session.generate_reply(
        instructions=(
            "Say exactly: "
            "Namaste, main Priya hoon, kaise help kar sakti hoon?"
        ),
    )


# ============================================================================
# WORKER
# ============================================================================

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            num_idle_processes=1,
            agent_name="priya",
        )
    )
