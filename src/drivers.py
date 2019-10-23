import numpy as np

class ExamineData():
    def __init__(self, data: np.array, data_name: str = ''):
        header = ''.join([data_name, '\n', '='*len(data_name)])
        print(f'{header}\n{data}')
        print(f'{data_name} shape: {data.shape}\n\n\n')
    def pause(self):
        input()

class CheckValidLength():
    def __init__(self, calculated, threshold, mode: str, strict: bool = True):
        if mode == 'less than':
            if strict:
                assert calculated < threshold, f'{calculated} >= {threshold} !'
            else:
                assert calculated <= threshold, f'{calculated} > {threshold} !'
        elif mode == 'greater than':
            if strict:
                assert calculated > threshold, f'{calculated} <= {threshold} !'
            else:
                assert calculated >= threshold, f'{calculated} < {threshold} !'
        elif mode == 'equal':
            assert calculated == threshold, f'{calculated} != {threshold} !'
        elif mode == 'not equal':
            assert calculated != threshold, f'{calculated} = {threshold} !'
        else:
            print('mode variable not valid! Validation skipped.')
