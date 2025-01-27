from numpy.f2py.crackfortran import endifs


def scale(a):
    low = min(a)
    high = max(a)

    ascale = []
    for i in a:
        num = i - low
        den = high - low
        ascale.append(num / den)

    return ascale

a = [1,2,3,4,5]


print('function answer is ', scale(a))