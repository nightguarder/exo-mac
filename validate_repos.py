import asyncio
import aiohttp
import sys
import os

# Add current directory to path so we can import exo
sys.path.append(os.getcwd())

from exo.models import model_cards

async def check_repo(session, model_id, engine, repo):
    url = f"https://huggingface.co/api/models/{repo}"
    try:
        async with session.head(url) as response:
            if response.status not in [200, 206]:
                print(f"❌ {model_id} [{engine}]: {repo} -> Status {response.status}")
                return False
            else:
                return True
    except Exception as e:
        print(f"⚠️ {model_id} [{engine}]: {repo} -> Error {e}")
        return False

async def main():
    headers = {"User-Agent": "Exo-Model-Validator"}
    async with aiohttp.ClientSession(headers=headers) as session:
        print(f"Checking {len(model_cards)} models...")
        tasks = []
        for model_id, card in model_cards.items():
            repos = card.get("repo", {})
            for engine, repo in repos.items():
                tasks.append(check_repo(session, model_id, engine, repo))
        
        results = await asyncio.gather(*tasks)
        failures = len([r for r in results if not r])
        print(f"\nDone. {failures} failures found out of {len(tasks)} repos checked.")

if __name__ == "__main__":
    asyncio.run(main())
