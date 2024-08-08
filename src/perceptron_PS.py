
#x_input = [0.1,0.5,0.2]
x_input = [0.5,0.7,0.2]
w_weights = [0.4,0.3,0.6]
threshold = 0.5

def step_function(weighted_sum):
    if weighted_sum > threshold:
        return 1
    else:
        return 0

def perceptron():
    weighted_sum = 0
    count = 0 
    for x,w in zip(x_input, w_weights):
        count +=1
        weighted_sum += x*w
        print("[" + str(count) + "] weighted sum= " + str(weighted_sum))
    return step_function(weighted_sum)

output = perceptron()
print("[*] output= " + str(output))