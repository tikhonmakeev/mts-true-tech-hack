class EmotionAgent:
    def __init__(self):
        pass

    def detect_emotion(self, summary):
        if "!" in summary:
            return "anger"
        return "neutral"
