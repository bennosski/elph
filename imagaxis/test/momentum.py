import src
from convolution import conv
from numpy import *

def test_conv1(N):
    x = random.randn(N)
    y = random.randn(N)
    z = zeros(N)

    for ik in range(N):
        for iq in range(N):
            z[ik] += x[iq] * y[((ik-iq)+N//2)%N]

    w = conv(x, y, ['q,k-q'], [0], [True], None, kinds=(None,None,None))

    print(amax(abs(w-z)))


def test_conv2(N):
    x = random.randn(N)
    y = random.randn(N)
    z = zeros(N)

    for ik in range(N):
        for iq in range(N):
            z[ik] += x[iq] * y[((ik+iq)-N//2)%N]

    w = conv(x, y, ['q,k+q'], [0], [True], None, kinds=(None,None,None))

    print(amax(abs(w-z)))


test_conv1(4)
test_conv2(4)
