# Common Libraries
import time

# Sources

def time_decorator(name):
    def callf(func):
        def inner(*args, **kwargs):
            print("%s Start" % name)
            start_time = time.time()
            func(*args, **kwargs)
            print("%s End (%.2f seconds) " % (name, time.time() - start_time))
        return inner
    return callf

def time_decorator_for_return(name):
    def callf(func):
        def inner(*args, **kwargs):
            print("%s Start" % name)
            start_time = time.time()
            returnValue = func(*args, **kwargs)
            print("%s End (%.2f seconds) " % (name, time.time() - start_time))
            return returnValue
        return inner
    return callf

if __name__ == "__main__":
    @time_decorator_for_return('리턴값이 있는 함수')
    def func_for_return(a, b):
        time.sleep(1)
        return a+b

    @time_decorator('리턴값이 없는 함수')
    def func():
        c = func_for_return(29, 33)
        time.sleep(1.5)
        print('리턴값 : %d' % c)
        retu
    func()