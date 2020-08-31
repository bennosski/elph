

# memory estimate

#nk = 20
#nw = 512
#nr = 8200

nk = 128
nw = 256
nr = 8.4 / 0.0025
#nr = 1680


# S, S0, PI, PI0, G, GG, D
# fac = 14
# sc = 4

memi = 16*nk*nk*2*nw*14*4 / 1e9
print('memory imag = ', memi, 'GB')

# SR, SR0, PIR, PIR0, GR, DR, Gsum+, Gsum-

memr = 16*nk*nk*nr*14*4 / 1e9
print('memory real = ', memr, 'GB')
