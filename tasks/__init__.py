def get_task(name, file=None):
    if name == 'trivia_creative_writing':
        from .trivia_creative_writing import TriviaCreativeWritingTask
        return TriviaCreativeWritingTask(file)
    elif name == 'logic_grid_puzzle':
        from .logic_grid_puzzle import LogicGridPuzzleTask
        return LogicGridPuzzleTask(file)
    elif name == 'codenames_collaborative':
        from .codenames_collaborative import CodenamesCollaborativeTask
        return CodenamesCollaborativeTask(file)
    elif name == 'hf_numeric_math':
        from .hf_numeric_math import HFNumericMathTask
        return HFNumericMathTask(file)
    elif name == 'hf_multiple_choice':
        from .hf_multiple_choice import HFMultipleChoiceTask
        return HFMultipleChoiceTask(file)
    else:
        raise NotImplementedError