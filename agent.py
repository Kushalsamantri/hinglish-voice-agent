import os
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, rime, silero

load_dotenv()

async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    assistant = VoiceAssistant(
        vad=silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.4,
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="multi",
            smart_format=True,
            keywords=[
                ("order", 2.0),
                ("track", 2.0),
                ("delivery", 2.0),
                ("status", 2.0),
            ],
        ),

        llm=openai.LLM(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        ),

        tts=rime.TTS(
            model="arcana",
            speaker="celeste",
        ),

        chat_ctx=llm.ChatContext().append(
            role="system",
            text=(
                "You are a friendly Indian customer support agent for an e-commerce platform. "
                "Speak naturally in Hinglish and match the user's language. "
                "Use English, Hindi, or a natural mix of both. "
                "Always write Hindi using the Latin alphabet, never Devanagari. "
                "Keep responses short, ideally 1-2 sentences. "
                "If the user's speech appears unclear or corrupted, politely ask them to repeat."
            )
        ),
    )

    assistant.start(ctx.room)

    await assistant.say(
        "Namaste! How can I help you today?",
        allow_interruptions=True,
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))