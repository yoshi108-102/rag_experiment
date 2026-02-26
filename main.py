import collections
from dotenv import load_dotenv
from src.agents.gate import analyze_input
from src.routing.router import execute_route
from src.core.chat_logging import ChatSessionLogger

# Load environment variables from .env file
load_dotenv()

INITIAL_ASSISTANT_GREETING = "お疲れ様です！本日の作業はどうでしたか？何か気になったことや、迷った瞬間はありましたか？"


def main():
    chat_logger = ChatSessionLogger.create(app_name="cli")
    llm_context = collections.deque(maxlen=10)

    print("Welcome to Reflective Gate Chat!")
    print("Type 'exit' or 'quit' to stop.\n")
    
    print(f"AI:   {INITIAL_ASSISTANT_GREETING}\n")
    chat_logger.log_message("assistant", INITIAL_ASSISTANT_GREETING, source="initial_greeting", message_index=0)
    llm_context.append({"role": "assistant", "content": INITIAL_ASSISTANT_GREETING})
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                chat_logger.log_event("conversation_closed_by_user")
                break
                
            if not user_input.strip():
                continue

            llm_context.append({"role": "user", "content": user_input})
            chat_logger.log_message("user", user_input)
                
            # Step 1: Analyze input using the Gate Model
            decision, reasoning = analyze_input(user_input, list(llm_context))
            
            # Step 2: Execute the routing logic Based on the decision
            response = execute_route(decision)
            llm_context.append({"role": "assistant", "content": response})
            chat_logger.log_message(
                "assistant",
                response,
                debug_info={
                    "route": decision.route,
                    "reason": decision.reason,
                    "reasoning": reasoning,
                },
            )
            
            print(f"AI:   {response}\n")
            
            if decision.route == "FINISH":
                chat_logger.log_event("conversation_finished", route=decision.route, reason=decision.reason)
                continue
            
        except KeyboardInterrupt:
            chat_logger.log_event("conversation_interrupted")
            print("\nExiting...")
            break
        except Exception as e:
            chat_logger.log_error(e, stage="cli_loop")
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()
