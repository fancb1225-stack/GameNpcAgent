from __future__ import annotations

from db.db_init import init_database
from model.dialogue_model import NpcReplyCreate, PlayerMessageCreate
from model.relationship_model import RelationshipDelta
from service.cafe_scene_service import CafeSceneService
from service.dialogue_service import DialogueService


if __name__ == "__main__":
    init_database()

    scene_service = CafeSceneService()
    result = scene_service.enter_morning_cafe(player_id="p001", nickname="玩家A", session_id="demo_session_001")
    print("遇见 NPC:", [npc["name"] for npc in result["npcs"]])

    dialogue_service = DialogueService(scene_service.db)
    dialogue_service.add_player_message(
        PlayerMessageCreate(
            session_id="demo_session_001",
            player_id="p001",
            npc_id="barista_001",
            content="今天推荐什么咖啡？",
        )
    )
    dialogue_service.add_npc_reply(
        NpcReplyCreate(
            session_id="demo_session_001",
            player_id="p001",
            npc_id="barista_001",
            content="今天小雨，热拿铁会更舒服；如果你想要清亮一点，埃塞俄比亚手冲也不错。",
            intent="ask_recommendation",
            action="recommend_coffee",
            relationship_delta=RelationshipDelta(familiarity=1, fondness=2),
        )
    )
    print("对话条数:", len(dialogue_service.list_messages("demo_session_001")))
    scene_service.close()
