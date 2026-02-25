import sys
from dotenv import load_dotenv
from src.agents.gate import analyze_input
from src.routing.router import execute_route

# Load environment variables from .env file
load_dotenv()

def main():
    print("Welcome to Reflective Gate Chat!")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.strip().lower() in ['exit', 'quit']:
                break
                
            if not user_input.strip():
                continue
                
            # Step 1: Analyze input using the Gate Model
            decision = analyze_input(user_input)
            
            # Step 2: Execute the routing logic Based on the decision
            response = execute_route(decision)
            
            print(f"AI:   {response}\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[Error]: {e}")

if __name__ == "__main__":
    main()

