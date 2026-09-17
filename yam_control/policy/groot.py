'''현재 보류된 GR00T backend의 명시적인 실패 경계를 제공한다.'''


class GrootBackend:
    '''GR00T가 public type에는 존재하지만 아직 실행되지 않도록 차단한다.'''

    def __init__(
        self,
    ) -> None:
        '''GR00T inference가 구현되지 않았음을 즉시 알린다.'''
        raise NotImplementedError('GR00T inference is not implemented yet')
