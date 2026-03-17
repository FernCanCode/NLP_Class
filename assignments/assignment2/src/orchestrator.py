'''
Orchestrator Logic Plan
1. Initialize the debate agents
2. The Debate Loop:
    a. Round Robin
    b. Transcript passing
    c. adaptive stopping
3. Judge Evaluation
4. Output logging and evaluation
'''

class DebateManager:
    def __init__(self, proponent, opponent, judge):
        self.proponent = proponent
        self.opponent = opponent
        self.judge = judge
        self.full_transcript = []
        self.consecutive_arguments = 0
    #use a simple check like .lower() or a regular expression to look for the words "Yes" or "No" within the agent's <argument> tags
    #return the verdict as a string
    def parse_verdict(self, text):
        if not text:
            return None
        text = text.lower()
        if "final answer: yes" in text or "verdict: yes" in text:
            return "Yes"
        elif "final answer: no" in text or "verdict: no" in text:
            return "No"
        else:
            return None
    
    # Calls generate_response for both agents independently with no previous transcript
    # Logic:
    #   gets ans_a and ans_b
    #   uses parse_verdict to compare them
    #   if they match, marks it as consensus and returns True to tell the main loop to skip phase 2
    #   if they don't match, returns False to tell the main loop to proceed to phase 2
    def run_phase_1(self, question):
        ans_a = self.proponent.generate_response(question)
        ans_b = self.opponent.generate_response(question)
        
        self.full_transcript.append({"round": 0, "role": "Proponent", "content": ans_a})
        self.full_transcript.append({"round": 0, "role": "Opponent", "content": ans_b})
        
        if self.parse_verdict(ans_a) == self.parse_verdict(ans_b):
            return True
        else:
            return False
    
    # The Multi-Round Debate Loop
    # Logic:
    #   Starts a loop for N rounds ( N >= 3)
    #   Turn A: Calls proponent.generate_response(). The input is the last opponents response
    #   Turn B: Calls opponent.generate_response(). The input is the proponent's just generated response
    #   Logging: Append both responses to self.full_transcript in the correct order
    #   Adaptive Stopping: Uses parse_verdict on both. If they agree for two consecutive rounds, break the loop
    def run_phase_2(self, question, max_rounds=3):
        last_opponent_arg = None
        for round in range(max_rounds):
            print(f"--- Round {round + 1} ---")
            # 1. Proponent's Turn. Hear what opponent last said (if anything)
            prop_response = self.proponent.generate_response(question, opponent_argument=last_opponent_arg)
            self.full_transcript.append({"round": round, "role": "Proponent", "content": prop_response})
            # 2. Opponent's Turn. Respond directly to what Proponent just said
            opp_response = self.opponent.generate_response(question, opponent_argument=prop_response)
            self.full_transcript.append({"round": round, "role": "Opponent", "content": opp_response})
            # 3. Adaptive Stopping Check
            # Extract Yes/No from both to see if they've started to agree
            prop_verdict = self.parse_verdict(prop_response)
            opp_verdict = self.parse_verdict(opp_response)

            if prop_verdict == opp_verdict and prop_verdict is not None:
                self.consecutive_arguments += 1
            else:
                self.consecutive_arguments = 0

            # If we have 2 consecutive rounds of agreement, break the loop
            if self.consecutive_arguments >= 2:
                print(f"Adaptive Stopping triggered: Consensus reached at round {round + 1}")
                break

    # Formats the final transcript and calls the Judge to render a verdict
    def run_phase_3(self, question):
        
        # 1. Format the full transcript for the Judge's prompt
        formatted_transcript = ""
        for entry in self.full_transcript:
            role = entry["role"]
            content = entry["content"]
            formatted_transcript += f"\n--- {role} ---\n{content}\n"

        print("--- Rendering Final Judgment ---")

        # 2. Call the Judge Agent
        # Uses {{QUESTION}} and {{TRANSCRIPT}} placeholders in judge.txt prompt template
        judge_response = self.judge.generate_response(
            question=question,
            custom_transcript=formatted_transcript
        )

        return judge_response

    def run_full_debate(self, question, max_rounds=3):
        
        # Phase 1: Initial Check
        consensus_reached = self.run_phase_1(question)

        # Phase 2: If no consensus, run the debate loop
        if not consensus_reached:
            self.run_phase_2(question, max_rounds=max_rounds)
        else:
            print("Initial consensus reached. Skipping debate loop")
        
        # Phase 3: Final Judgement based on the full history
        final_verdict = self.run_phase_3(question)
        
        return final_verdict

    def reset(self):
        self.full_transcript = []
        self.consecutive_arguments = 0

        self.proponent.messages = []
        self.opponent.messages = []
        self.judge.messages = []

        self.proponent.system_prompt = ""
        self.opponent.system_prompt = ""
        self.judge.system_prompt = ""
    
    def extract_confidence(self, judge_text):
        # Use regex to extract the confidence score
        import re
        #Searches for "Confidence Score:" followed by a digit 1-5
        match = re.search(r"confidence score:\s*([1-5])", judge_text.lower())
        if match:
            return int(match.group(1))
        else:
            return None
        