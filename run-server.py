import asyncio
import logging
from main import run_live_analysis
from config import Config


async def run_server(interval):
    while True:
        try:
            await run_live_analysis()  # Run the live analysis
        except Exception as e:
            logging.error(f"An error occurred during live analysis: {str(e)}")

        # Wait for the specified interval
        await asyncio.sleep(interval)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Set the desired interval in seconds
    interval = Config.SERVER_INTERVAL  # Example: Run the analysis every 60 seconds

    # Run the server
    asyncio.run(run_server(interval))
