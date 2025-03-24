'''
    L1       L2
#--------|--------#
# agent  | apple  #
#--------|--------#
'''

V = {"L1": 0.0, "L2": 0.0}
p = 0.5
gamma = 0.9

cnt = 0
while True:
    t = p * (-1 + gamma*V["L1"]) + p * (1 + gamma*V["L2"])
    delta = abs(t - V["L1"])
    V["L1"] = t
    
    t = p * (0 + gamma*V["L1"]) + p * (-1 + gamma*V["L2"])
    delta = abs(t - V["L2"])
    V["L2"] = t
    
    cnt += 1
    if delta < 0.0001:
        print(V)
        print("count of updates: ",cnt)
        break