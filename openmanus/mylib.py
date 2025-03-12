
import asyncio

from app.agent.manus import Manus

async def run_agent(prompt: str):
    """Agent执行方法"""
    try:
        agent = Manus()
        await agent.run(prompt)
        return True
    except Exception as e:
        return False
    