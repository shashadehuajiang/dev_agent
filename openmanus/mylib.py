
import asyncio

from app.agent.manus import Manus

async def run_agent(prompt: str, max_steps = 50):
    """Agent执行方法"""
    try:
        agent = Manus(max_steps = max_steps)
        await agent.run(prompt)
        return True
    except Exception as e:
        return False
    