import asyncio
from dotenv import load_dotenv
from livekit.plugins import rime
from livekit.agents.utils.http_context import open as open_http_context

load_dotenv()

async def main():
    async with open_http_context():
        tts = rime.TTS(
            model="coda",
            speaker="lyra",
            use_websocket=False,
        )

        stream = tts.synthesize(
            "Namaste! Main Priya hoon. Aapki kya help kar sakti hoon?"
        )

        count = 0

        async for frame in stream:
            count += 1
            print(
                    "FRAME",
                    count,
                    "samples=", frame.frame.samples_per_channel,
                    "channels=", frame.frame.num_channels,
                    "rate=", frame.frame.sample_rate,
                )

        print("TOTAL FRAMES:", count)

        await tts.aclose()

if __name__ == "__main__":
    asyncio.run(main())