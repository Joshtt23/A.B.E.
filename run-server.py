import asyncio
import logging
import time
from main import perform_live_analysis
from config import Config


async def run_server(interval):
    while True:
        try:
            logging.info("Starting live analysis...")
            start_time = time.time()

            await perform_live_analysis()  # Run the live analysis

            end_time = time.time()
            execution_time = end_time - start_time
            logging.info(f"Live analysis completed in {execution_time} seconds.")
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
