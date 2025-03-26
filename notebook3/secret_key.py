import random
def generate_secret_key(Slots,HammingWeight=32):
    secretKey = [0]*(2*Slots)
    count = 0
    while count < HammingWeight:
        pos = random.randint(0,2*Slots-1)
        if secretKey[pos] != 0:
            continue
        val = random.randint(0,1)
        if val == 0:
            secretKey[pos] = -1
        elif val == 1:
            secretKey[pos] = 1
        else:
            raise Exception("")
        count += 1
    return secretKey