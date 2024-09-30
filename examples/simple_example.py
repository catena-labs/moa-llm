import asyncio

from moa_llm import create_moa_from_config

async def main():
    moa = create_moa_from_config('examples/openai_compatible_server/moa_config.yaml', is_file_path=True)
    result = await moa.process("Write a function that takes a list of numbers and returns the sum of the squares of the numbers.")
    print(result)

asyncio.run(main())