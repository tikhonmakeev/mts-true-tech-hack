class QualityAssuranceAgent:
    def __init__(self):
        pass

    def check_quality(self, summary):
        if "необходимо" in summary:
            return "Предложите извинения и уточните требования клиента"
        return "Диалог соответствует стандартам"
