'''
    L1       L2
#--------|--------#
# agent  | apple  #
#--------|--------#
'''
V = {"L1": 0.0, "L2": 0.0}
new_V = V.copy()
p = 0.5
gamma = 0.9

v_L1 = 0.0
v_L2 = 0.0

cnt = 0
while True:
    new_V["L1"] = p*(-1 + gamma * V["L1"]) + p*(1 + gamma * V["L2"])
    new_V["L2"] = p*(0 + gamma * V["L1"]) + p*(-1 + gamma * V["L2"])
    
    delta = abs(new_V["L1"] - V["L1"])
    delta = max(delta, abs(new_V["L2"] - V["L2"]))
    
    V = new_V.copy()
    
    cnt += 1
    if delta < 0.0001:
        print(V)
        print("count of updates: ", cnt)
        break