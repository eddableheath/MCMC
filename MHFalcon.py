"""
An attempt to implement the Metropolis-Hastings sampler into FALCON and generally to test and get the hang
of Thomas Prest's FALCON code
"""

from FALCON import falcon as fn

sk = fn.SecretKey(8)
pk = fn.PublicKey(sk)

m = 'can it be a string?'
sig = sk.sign(m)

print(sk)
print(pk)
print(sig)
print(pk.verify(m,sig))
