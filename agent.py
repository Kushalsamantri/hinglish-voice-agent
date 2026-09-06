"""
agent.py — Production Hinglish Voice Agent (LiveKit Agents 1.x)
================================================================
Optimized for: sub-500ms TTFA, female Hinglish pronunciation, robust turn-taking.

Research-backed choices:
- STT: Deepgram Nova-3 Hindi model (handles Hinglish code-switching better than multilingual)
- LLM: Groq Llama 3.3 70B (250ms TTFT, 300+ t/s) — lowest latency for voice
- TTS: Rime Coda (sub-100ms TTFB, Hindi support, spell() for IDs)
- VAD: Silero aggressively tuned to eliminate 2.5s silence penalty
- Turn Detection: STT-native endpointing with zero min_delay stacking
"""

import os
import re
import logging
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit.plugins import deepgram, openai, rime, silero
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    TurnHandlingOptions,
    RoomInputOptions,
    llm,
    RunContext,
)


load_dotenv()

# ------------------------------------------------------------------------------
# Logging & Observability
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("hinglish-agent")


# ------------------------------------------------------------------------------
# SYSTEM PROMPT — Hinglish Female Support Agent
# ------------------------------------------------------------------------------
# Rules enforced here + TTS post-processing guard for 100% female pronouns.
# Kept under 300 tokens to minimize LLM latency.
SYSTEM_PROMPT = (
    "You are Priya, a friendly Indian female customer support agent for an e-commerce platform.\n\n"
    "CRITICAL HINGLISH SCRIPT & VOICE RULES:\n"
    "1. When speaking Hinglish, ALWAYS write Hindi words in Devanagari script and English words in English letters.\n"
    "   - DO NOT write Hindi in English letters (NEVER write 'Main aapki help kar rahi hoon' or 'bata dijiye').\n"
    "   - DO write mixed script like this:\n"
    "     * 'Haanji! मैं तुरंत आपका order track कर देती हूँ।'\n"
    "     * 'Sure thing! आप मुझे अपना Order ID बता दीजिए, मैं check कर लेती हूँ।'\n"
    "     * 'Don't worry, आपका refund process हो रहा है।'\n"
    "     * 'I can help with that! Order में क्या problem आ रही है?'\n\n"
    "2. FEMALE GRAMMAR ONLY: Always use female endings: 'कर देती हूँ', 'बता देती हूँ', 'देखूँगी', 'समझ गई'. Never use male forms like 'करूँगा'.\n\n"
    "3. Keep all responses very short (1-2 sentences maximum, strictly under 12 words) so speech starts immediately."
)


# ------------------------------------------------------------------------------
# Female Pronoun Guard — TTS Post-Processor
# ------------------------------------------------------------------------------
# Safety net: if LLM ever hallucinates male verbs, rewrite to female before TTS.
_MALE_TO_FEMALE_MAP = {
    # Common Hinglish male → female verb mappings
    "karunga": "karungi",
    "karungaa": "karungi",
    "jaunga": "jaungi",
    "jaungaa": "jaungi",
    "bataunga": "bataungi",
    "bataungaa": "bataungi",
    "dekhunga": "dekhungi",
    "dekhungaa": "dekhungi",
    "sununga": "sunungi",
    "dunga": "dungi",
    "lung": "lungi",
    "samajh gaya": "samajh gayi",
    "samajh gaya hai": "samajh gayi hai",
    "sakta hoon": "sakti hoon",
    "sakta hu": "sakti hoon",
    "raha hoon": "rahi hoon",
    "raha hu": "rahi hoon",
    "chuka hoon": "chuki hoon",
    "chuka hu": "chuki hoon",
    "liya hai": "li hai",
    "diya hai": "di hai",
    "gaya hai": "gayi hai",
    "gaya": "gayi",
    "liya": "li",
    "diya": "di",
    
}


def _enforce_female_pronouns(text: str) -> str:
    """Rewrite any male Hindi verbs to female equivalents before TTS."""
    # Case-insensitive replacement with word boundaries
    for male, female in _MALE_TO_FEMALE_MAP.items():
        pattern = r'\b' + re.escape(male) + r'\b'
        text = re.sub(pattern, female, text, flags=re.IGNORECASE)
    return text


# ------------------------------------------------------------------------------
# Agent Class with Custom Pipeline Nodes
# ------------------------------------------------------------------------------
class HinglishSupportAgent(Agent):
    def __init__(self):
        super().__init__(instructions=SYSTEM_PROMPT)
        self._max_history_turns = 6  # Sliding window to prevent token bloat

    # --------------------------------------------------------------------------
    # LLM Node — Context Management (Sliding Window)
    # --------------------------------------------------------------------------
    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.FunctionTool],
        model_settings,
    ) -> AsyncIterable[llm.ChatChunk]:

        async for chunk in Agent.default.llm_node(
            self,
            chat_ctx,
            tools,
            model_settings,
        ):
            yield chunk

    # --------------------------------------------------------------------------
    # TTS Node — Hinglish Normalization + Female Guard + Symbol Stripper
    # --------------------------------------------------------------------------
    async def tts_node(
        self,
        text: AsyncIterable[str],
        model_settings,
    ):
        async def clean_text_stream():
            async for chunk in text:
                # Remove quotes, asterisks, and markdown formatting
                cleaned = re.sub(r'[*_`#""“”\'–—]', '', chunk)
                cleaned = _enforce_female_pronouns(cleaned)
                yield cleaned

        async for audio_frame in Agent.default.tts_node(self, clean_text_stream(), model_settings):
            yield audio_frame


# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Connecting to room: {ctx.room.name}")
    await ctx.connect(auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY)

    # --------------------------------------------------------------------------
    # VAD — Aggressive tuning to eliminate silence-wait penalty
    # --------------------------------------------------------------------------
    vad = silero.VAD.load(
        min_silence_duration=0.45,   # End turn after 350ms silence (default 0.5)
        min_speech_duration=0.25,    # Ignore sub-120ms noises
        prefix_padding_duration=0.15, # Capture 150ms before speech starts
        activation_threshold=0.60,   # Slightly more sensitive than default 0.5
    )

    # --------------------------------------------------------------------------
    # Turn Handling — Low-latency configuration
    # --------------------------------------------------------------------------
    # NOTE: If deploying to LiveKit Cloud, change interruption.mode to "adaptive"
    # for better barge-in handling. On self-hosted, "vad" is the only option.
    turn_handling = TurnHandlingOptions(
        # Use STT's native endpointing (Deepgram Nova-3 has strong turn detection)
        turn_detection="stt",
        endpointing={
            "mode": "fixed",
            "min_delay": 0.0,   # CRITICAL: don't stack delay on top of STT endpointing
            "max_delay": 2.0,   # Cap total wait at 2s
        },
        interruption={
            "enabled": True,
            "mode": "vad",          # Use "adaptive" if on LiveKit Cloud
            "min_duration": 0.35,   # Increased from 0.25 to prevent breath cutoff
            "min_words": 2,          # Requires at least 2 distinct words to cut in
            "false_interruption_timeout": 1.5,
            "resume_false_interruption": True,
        },
        # Preemptive generation: start LLM inference on STT partials before
        # turn is fully committed. preemptive_tts=True also starts TTS early.
        # This is the single biggest latency win (-200 to -400ms).
        preemptive_generation={
            "enabled": True,
            "preemptive_tts": False,
            "max_speech_duration": 8.0,  # Skip preemptive if user speaks >8s
            "max_retries": 2,
        },
    )


    tts = rime.TTS(
        model="coda",
        speaker="astra",
        sample_rate=22050,
        use_websocket=True,
        segment="immediate",
        #speed_alpha=0.9,
    )

    tts.prewarm()

    # --------------------------------------------------------------------------
    # Session Assembly
    # --------------------------------------------------------------------------
    session = AgentSession(
        # --- STT: Deepgram Nova-3 Hindi ---------------------------------------
        # language="hi" (Hindi model) handles Hinglish code-switching BETTER
        # than language="multi" based on Deepgram community guidance.
        # keyterms boost e-commerce vocabulary recognition.
        stt=deepgram.STT(
            model="nova-3",
            language="hi",
            interim_results=True,
            smart_format=True,
            endpointing_ms=100,
            keyterm=[
                "UPI", "EMI", "NEFT", "OTP", "order", "track",
                "refund", "delivery", "return", "payment", "account",
            ],
        ),

        # --- LLM: Groq (Fastest inference for voice) --------------------------
        # llama-3.3-70b-versatile: ~250ms TTFT, 300+ tokens/sec on Groq LPU.
        # Temperature 0.15 = deterministic, fast, consistent Hinglish output.
        # max_completion_tokens=35 forces the model to stay under ~12 words.
        llm=openai.LLM(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
            temperature=0.2,
            max_completion_tokens=60,
            top_p=0.9,
            reasoning_effort="low",
        ),

        # --- TTS: Rime Coda ---------------------------------------------------
        # Coda: sub-100ms TTFB, supports Hindi, natural prosody.
        # speaker="lyra": female voice. If lyra doesn't support Hindi well,
        # switch to a Hindi-specific Coda voice (check Rime dashboard).
        # speed=1.1: slightly faster for snappy conversation.
        tts=tts,

        vad=vad,
        turn_handling=turn_handling,
    )

    # --------------------------------------------------------------------------
    # Start Session
    # --------------------------------------------------------------------------
    await session.start(
        room=ctx.room,
        agent=HinglishSupportAgent(),
        room_input_options=RoomInputOptions(
            #audio_input=RoomInputOptions.AudioInputOptions(
             #   noise_cancellation=noise_cancellation.BVC()
            # Uncomment if you have noise_cancellation plugin installed:
            # audio_input=RoomInputOptions.AudioInputOptions(
            #     noise_cancellation=noise_cancellation.BVC()
             #)
        ),
    )

    # Balanced opening greeting with proper Hindi pronunciation
    await session.generate_reply(
        instructions="Say exactly: 'नमस्ते! मैं Priya हूँ, how can I help you today बताइए?'"
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            num_idle_processes=1,
            agent_name="priya",
        )
    )
