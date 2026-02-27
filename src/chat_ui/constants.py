"""チャットUI層で共有する文言・RAG制御パラメータを定義する。"""

INITIAL_ASSISTANT_MESSAGE = (
    "お疲れ様です！本日の作業はどうでしたか？"
    "何か気になったことや、迷った瞬間はありましたか？"
)

RAG_STREAK_TRIGGER = 3
RAG_COOLDOWN_TURNS = 2
RAG_BUFFER_MAX_ITEMS = 6

RAG_ELIGIBLE_ROUTES = ("DEEPEN", "CLARIFY", "PARK", "FINISH")
BOUNDARY_ROUTES = {"PARK", "FINISH"}
