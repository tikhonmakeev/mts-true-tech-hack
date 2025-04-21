import json
import random

from fastapi import APIRouter


router = APIRouter(prefix="/api/v1/support")


@router.get("/random_user_request")
async def get_random_user_request() -> str:
    with open("app/data/support_requests_text_only.json", encoding="utf-8") as f:
        support_requests = json.load(f)
    return random.choice(support_requests)
