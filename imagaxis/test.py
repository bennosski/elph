
x = 0



class C:

    w = 8

    def f():
        z = 2
        y = 1
        print('my locals')
        print(locals())

        print('globals')
        print(globals())

        print(y)

C.f()


print(x)
